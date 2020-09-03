import torch
import torch.nn as nn
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
        x = self.merge_lin(torch.cat((x0, x1), dim=1))

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
        valid_mask = batch[:, -1].bool()
        x = x[valid_mask]

        return x

    def merge_batches(self, batch):
        """ mask out nodes that don't map to valid actions
        so that the actionable (empty cells) align
        """
        for i in range(2):
            # clear the nodes that don't represent valid actions
            valid_mask = batch[i][:, -1].bool()
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


class GATResBlock_2Bridge(torch.nn.Module):
    def __init__(self, channels, affine_bn=True, heads=1):
        super().__init__()
        self.gc_1 = GATConv(in_channels=channels, out_channels=channels, heads=heads)
        self.bn_1 = nn.BatchNorm1d(num_features=channels, affine=affine_bn)
        self.elu = nn.ELU()
        self.gc_2 = GATConv(in_channels=channels, out_channels=channels, heads=heads)
        self.bn_2 = nn.BatchNorm1d(num_features=channels, affine=affine_bn)

    def forward(self, x, edge_index, edge_index_2bridge):
        residual = x
        x = self.gc_1(x, edge_index)
        x = self.bn_1(x)
        x = self.elu(x)
        x = self.gc_2(x, edge_index_2bridge)
        x = self.bn_2(x).relu()
        x += residual

        return x.relu()


class Trunk_2Bridge(nn.Module):
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
            self.trunk_mlist.append(GATResBlock_2Bridge(channels=h2_sz, heads=attn_heads))

    def forward(self, x, edge_index, edge_index_2bridge):
        for layer in self.trunk_mlist:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            elif isinstance(layer, GATResBlock_2Bridge):
                x = layer(x, edge_index, edge_index_2bridge)
            else:
                x = layer(x)
        return x


class GraphNet_2Bridge(GraphNet):
    def __init__(self, args):
        super(GraphNet, self).__init__()

        self.node_size_in = args.num_channels
        h1_sz = self.node_size_in*args.expand_base
        h2_sz = self.node_size_in*(args.expand_base**2)

        self.trunk = Trunk_2Bridge(self.node_size_in, h1_sz, h2_sz, args.attn_heads, args.res_blocks)
        self.p_head = PolicyHead(channels=h2_sz, action_size=args.action_size)
        self.v_head = ValueHead(channels=h2_sz, attn_heads=args.readout_attn_heads)

    def forward(self, x):
        edge_index, node_attr, batch = x
        out = []
        for i in range(2):
            o = self.trunk(node_attr[i], edge_index[i], edge_index[i+2])
            o = self.align_nodes(o, batch[i])
            out.append(o)

        batch = self.merge_batches(batch)
        # the batch indicies should be consistant with both sets of nodes at this point
        assert batch.size(0) == out[0].size(0)
        assert batch.size(0) == out[1].size(0)

        p = self.p_head(out[0], out[1], batch)
        v = self.v_head(out[0], out[1], batch)

        return p, v
  


class PolicyHead_1Trunk(nn.Module):
    def __init__(self, channels, action_size):
        super().__init__()
        self.lin = nn.Linear(in_features=channels, out_features=channels)
        self.bn = nn.BatchNorm1d(num_features=channels)
        self.final_lin = nn.Linear(in_features=channels, out_features=1)
        self.action_size = action_size
        self.readout = nn.LogSoftmax(dim=1)

    def forward(self, x, batch):
        x = self.lin(x)
        x = self.bn(x).relu()
        x = self.final_lin(x)
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


class ValueHead_1Trunk(nn.Module):
    def __init__(self, channels, attn_heads):
        super().__init__()
        self.channels = channels
        self.lin = nn.Linear(in_features=self.channels, out_features=self.channels)
        self.bn = nn.BatchNorm1d(num_features=self.channels)
        self.final_lin = nn.Linear(in_features=channels, out_features=channels)
        self.mha = nn.MultiheadAttention(embed_dim=self.channels, num_heads=attn_heads)
        self.readout_lin = nn.Linear(in_features=self.channels, out_features=1)
        self.query = torch.ones((1, self.channels))

    def forward(self, x, batch):
        x = self.lin(x)
        x = self.bn(x).relu()
        x = self.final_lin(x)

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


class GraphNet_1Trunk(nn.Module):

    def __init__(self, args):
        super(GraphNet_1Trunk, self).__init__()

        self.node_size_in = args.num_channels
        h1_sz = self.node_size_in*args.expand_base
        h2_sz = self.node_size_in*(args.expand_base**2)

        self.trunk = Trunk(self.node_size_in, h1_sz, h2_sz, args.attn_heads, args.res_blocks)
        self.p_head = PolicyHead_1Trunk(channels=h2_sz, action_size=args.action_size)
        self.v_head = ValueHead_1Trunk(channels=h2_sz, attn_heads=args.readout_attn_heads)

    def forward(self, x):
        edge_index, node_attr, batch = x
        x = self.trunk(node_attr, edge_index)

        # clear the nodes that don't represent valid actions
        valid_mask = batch[:, -1].bool()
        batch = batch[valid_mask]
        x = x[valid_mask]
        
        p = self.p_head(x, batch)
        v = self.v_head(x, batch)

        return p, v


class ValueHead_SideNode(nn.Module):
    ''' This is a readout head for the value function that uses only embedding vectors 
    from the nodes containing the artificial player indicator side cells
    '''
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.lin1 = nn.Linear(in_features=4*channels, out_features=4*channels)
        self.lin2 = nn.Linear(in_features=4*channels, out_features=1)

    def forward(self, side_nodes):
        # concatenate side node features for each graph
        out = torch.cat(side_nodes, dim=1)
        out = self.lin1(out).relu()
        out = self.lin2(out).tanh()

        return out


class GraphNet_SideNode(GraphNet):
    def __init__(self, args):
        super(GraphNet, self).__init__()

        self.node_size_in = args.num_channels
        h1_sz = self.node_size_in*args.expand_base
        h2_sz = self.node_size_in*(args.expand_base**2)

        self.trunk = Trunk(self.node_size_in, h1_sz, h2_sz, args.attn_heads, args.res_blocks)
        self.p_head = PolicyHead(channels=h2_sz, action_size=args.action_size)
        self.v_head = ValueHead_SideNode(channels=h2_sz)

    def side_node_attr(self, node_in, trunk_out, batch):
        batch_sz = batch[-1, 0].item() + 1
        side_nodes = []
        # select the two side nodes for each player graph using the input features that distinguish the side nodes
        for i in (1, 2):
            side_mask = node_in[:, i].bool()
            assert side_mask.sum() == batch_sz
            side_nodes.append(trunk_out[side_mask])

        return side_nodes

    def forward(self, x):
        edge_index, node_attr, batch = x
        out = []
        side_nodes = []
        for i in range(2):
            o = self.trunk(node_attr[i], edge_index[i])
            side_nodes.extend(self.side_node_attr(node_attr[i], o, batch[i]))
            o = self.align_nodes(o, batch[i])
            out.append(o)

        batch = self.merge_batches(batch)
        # the batch indicies should be consistant with both sets of nodes at this point
        assert batch.size(0) == out[0].size(0)
        assert batch.size(0) == out[1].size(0)

        p = self.p_head(out[0], out[1], batch)
        v = self.v_head(side_nodes)

        return p, v
