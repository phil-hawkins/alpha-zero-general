import torch
import torch.nn as nn
from collections import OrderedDict
from torch_geometric.nn import GATConv

     
class GATResBlock(torch.nn.Module):
    def __init__(self, channels, affine_bn=True, heads=1):
        super().__init__()
        self.gc_1 = GATConv(in_channels=channels, out_channels=channels, heads=heads)
        self.bn_1 = nn.BatchNorm1d(num_features=channels, affine=affine_bn)
        self.elu = nn.ELU()
        self.gc_2 = GATConv(in_channels=channels, out_channels=channels, heads=heads)
        self.bn_2 = nn.BatchNorm1d(num_features=channels, affine=affine_bn)

    def forward(self, x, edge_index):
        residual = x
        x = self.gc_1(x, edge_index)
        x = self.bn_1(x)
        x = self.elu(x)
        x = self.gc_2(x, edge_index)
        x = self.bn_2(x).relu()
        x += residual
        
        return x.relu()


class HeadBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p0_lin = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.p0_bn = nn.BatchNorm1d(num_features=in_channels)
        self.p1_lin = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.p1_bn = nn.BatchNorm1d(num_features=in_channels)
        self.merge_lin = nn.Linear(in_features=in_channels*2, out_features=out_channels)

    def forward(self, x0, x1):
        x0 = self.p0_lin(x0)
        x0 = self.p0_bn(x0).relu()
        x1 = self.p1_lin(x1)
        x1 = self.p1_bn(x1).relu()
        x = self.merge_lin(torch.cat((x0,x1), dim=1))
        
        return x


class PolicyHead(HeadBase):
    def __init__(self, channels, action_size):
        super().__init__(in_channels=channels, out_channels=1)
        self.action_size = action_size
        self.readout = nn.LogSoftmax(dim=1)

    def forward(self, x0, x1, batch):
        x = super().forward(x0, x1)
        batch_sz = batch[-1, 0].item() + 1
        indicies = batch[:, :2].T
        x = x.squeeze(dim=1)
        out = torch.sparse_coo_tensor(
            size=(batch_sz, self.action_size), 
            indices=indicies, 
            values=x, 
            device=x.device
        ).to_dense()

        return self.readout(out)


class ValueHead(HeadBase):
    def __init__(self, channels, attn_heads):
        super().__init__(channels, channels)
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads)
        self.readout_lin = nn.Linear(in_features=channels, out_features=1)
        self.query = torch.ones((1, channels))

    def forward(self, x0, x1, batch):
        x = super().forward(x0, x1)

        _, batch_counts = batch[:, 0].unique(sorted=True, return_counts=True)
        batch_sz = batch[-1, 0].item() + 1
        out = torch.empty((batch_sz, self.channels), dtype=torch.float, device=x.device)
        query = torch.ones((1, 1, self.channels), dtype=torch.float, device=x.device)
        batch_start_ndx = 0

        # do the readout attention on each graph seperatly
        for i, bc in enumerate(batch_counts):
            graph_batch = x[batch_start_ndx:batch_start_ndx + bc].unsqueeze(dim=0)
            batch_start_ndx += bc
            out[i], _ = self.mha(query, graph_batch, graph_batch, need_weights=False)
        out = self.readout_lin(out).tanh()

        return out


class Trunk(nn.Module):
    def __init__(self, node_size_in, h1_sz, h2_sz, attn_heads, res_blocks):
        super().__init__()

        self.trunk_mlist = nn.ModuleList([
            GATConv(in_channels=node_size_in, out_channels=h1_sz, heads=attn_heads), 
            nn.BatchNorm1d(num_features=h1_sz),
            nn.ELU(),
            GATConv(in_channels=h1_sz, out_channels=h2_sz, heads=attn_heads), 
            nn.BatchNorm1d(num_features=h2_sz),
            nn.ReLU()
        ])
        for i in range(res_blocks):
            self.trunk_mlist.append(GATResBlock(channels=h2_sz, heads=attn_heads))

    def forward(self, x, edge_index):
        for layer in self.trunk_mlist:
            if isinstance(layer, (GATConv, GATResBlock)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GraphNet(nn.Module):
    def __init__(self, args): 
        super(GraphNet, self).__init__()

        self.node_size_in = args.num_channels
        h1_sz = self.node_size_in*args.expand_base
        h2_sz = self.node_size_in*(args.expand_base**2)

        self.trunk = Trunk(self.node_size_in, h1_sz, h2_sz, args.attn_heads, args.res_blocks)
        self.p_head = PolicyHead(channels=h2_sz, action_size=args.action_size)
        self.v_head = ValueHead(channels=h2_sz, attn_heads=args.readout_attn_heads)

    def align_nodes(self, x, batch):
        # mask out nodes that don't map to valid actions
        valid_mask = batch[:,-1].bool()
        x = x[valid_mask]

        return x

    def merge_batches(self, batch):
        """ mask out nodes that don't map to valid actions
        so that the actionable (empty cells) align
        """
        for i in range(2):
            # clear the nodes that don't represent valid actions
            valid_mask = batch[i][:,-1].bool()
            batch[i] = batch[i][valid_mask]

        # the only difference in the batch mappings is the non-empty nodes for each player
        # so both batches should now map to the same nodes
        assert torch.equal(batch[0], batch[1])

        return batch[0]

    def forward(self, x):
        edge_index, node_attr, batch = x
        out = []
        for i in range(2):
            o = self.trunk(node_attr[i], edge_index[i])
            o = self.align_nodes(o, batch[i])
            out.append(o)

        batch = self.merge_batches(batch)
        # the batch indicies should be consistant with both sets of nodes at this point
        assert batch.size(0) == out[0].size(0)
        assert batch.size(0) == out[1].size(0)

        p = self.p_head(out[0], out[1], batch)
        v = self.v_head(out[0], out[1], batch)

        return p, v

class NativeGraphNet(GraphNet):
    """ accepts input natively in graph format rather than a grid tensor 
    """
    def to_player_graphs(self, x):
        # TODO
        bg = BoardGraph.from_grid_board(x)

        return edge_index, node_attr, batch