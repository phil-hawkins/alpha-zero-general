import numpy as np
import torch
import math

class Board():
    def __init__(self, np_pieces):
        assert np_pieces.size(0) == np_pieces.size(1)
        self.np_pieces = np_pieces

    @property
    def display_string(self):
        display_chars = {
            0 : ".", 
            -1 : "H", 
            1 : "V"
        }

        board_str = "`  "
        for c in range(self.np_pieces.shape[1]):
            board_str += "{}   ".format(c)
        board_str += "\n  "
        for c in range(self.np_pieces.shape[1]):
            board_str += "----"
        for r in range(self.np_pieces.shape[0]):
            row_chr = chr(ord("a") + r)
            board_str += "\n{}{} ` ".format(r*"  ", row_chr)
            for c in range(self.np_pieces.shape[1]):
                board_str += "  {} ".format(display_chars[self.np_pieces[r,c].item()])
            board_str += "`"
        board_str += "\n    {}".format(r*"  ")
        for c in range(self.np_pieces.shape[1]):
            board_str += "----"

        return board_str

    @property
    def size(self):
        return self.np_pieces.size(0)

    @property
    def cell_count(self):
        return self.np_pieces.numel()



class BoardGraph():
    def __init__(self, node_attr, edge_index, action_map):
        self.node_attr = node_attr
        self.edge_index = edge_index
        self.action_map = action_map
        self.device = node_attr.device

    def __str__(self):
        return "Node Attributes:\n{}\n\nAdjacency Matrix:\n{}\n\nBoard Map: {}\n".format(self.node_attr, self.adjacency_matrix.to_dense(), self.action_map)

    @classmethod
    def random_graph(cls, max_diameter, device):
        """ the graph is built from a standard hex board of max_diameter size but with up to 4 nodes merged with neighbours
        """
        g = cls.graph_from_board(Board(torch.zeros((max_diameter, max_diameter), device=device)))

        r, c = randint(0, self.max_diameter-1), randint(0, self.max_diameter-1)
        mcells = set()
        mcells.add((r, c))
        mcells.add((c, r))
        mcells.add((self.max_diameter-r, self.max_diameter-c))
        mcells.add((self.max_diameter-c, self.max_diameter-r))

        for cell in mcells:
            cell_ndx = cell[0] * self.max_diameter + cell[1]

    @classmethod
    def graph_from_board(cls, board):
        """ create directed edge index for a standard hex board

        returns: 
            graph object
        """
        device = board.np_pieces.device
        k = torch.tensor([
            [-1, -1,  0,  1,  1,  0],
            [ 0,  1,  1,  0, -1, -1]]
        , device=device).unsqueeze(dim=1).expand(2, board.cell_count, 6).reshape(2, -1)
        c = torch.ones((board.size, board.size), device=device).to_sparse().indices()
        c = c.unsqueeze(dim=2).expand(2, board.cell_count, 6).reshape(2, -1)
        edge_index = torch.cat([c, c + k], dim=0)

        # remove off-board edges
        mask = (edge_index.min(dim=0)[0] >= 0) & (edge_index.max(dim=0)[0] < board.size)
        edge_index = edge_index[:, mask].view(2, 2, -1)
        edge_index = edge_index[:, 0] * board.size + edge_index[:, 1]

        # add edges to left and right side nodes
        next_node_ndx = board.size**2
        left_edges = torch.stack([
            torch.full((board.size,), fill_value=next_node_ndx, dtype=torch.long, device=device),
            torch.arange(board.size, dtype=torch.long, device=device) * board.size
        ])
        right_edges = torch.stack([
            torch.full((board.size,), fill_value=next_node_ndx+1, dtype=torch.long, device=device),
            torch.arange(board.size, dtype=torch.long, device=device) * board.size + board.size-1
        ])
        top_edges = torch.stack([
            torch.full((board.size,), fill_value=next_node_ndx+2, dtype=torch.long, device=device),
            torch.arange(board.size, dtype=torch.long, device=device)
        ])
        bottom_edges = torch.stack([
            torch.full((board.size,), fill_value=next_node_ndx+3, dtype=torch.long, device=device),
            torch.arange(board.size, dtype=torch.long, device=device) + (board.size * (board.size-1))
        ])
        side_edge_index = torch.cat([left_edges, right_edges, top_edges, bottom_edges], dim=1)
        edge_index = torch.cat([edge_index, side_edge_index, side_edge_index[[1,0]]], dim=1)
        board_node_attr = torch.zeros((board.np_pieces.numel(), 3), dtype=torch.long, device=device)
        board_node_attr[:, 0] = board.np_pieces.flatten()
        side_node_attr = torch.tensor([
            [-1, 1, 0], # player -1 (H) side 1
            [-1, 0, 1], # player -1 (H) side 2
            [1, 1, 0],  # player 1 (V) side 1
            [1, 0, 1]   # player 1 (V) side 2
        ], dtype=torch.long, device=device)

        node_attr = torch.cat([board_node_attr, side_node_attr])
        action_map = torch.stack([
            torch.arange(node_attr.size(0), device=device),  # board_cell
            (node_attr[:, 0] == 0).long()                    # valid_action (1/0)
        ], dim=1)

        return cls(node_attr, edge_index, action_map)

    @property
    def adjacency_matrix(self):
        v = torch.ones_like(self.edge_index[0])
        return torch.sparse_coo_tensor(indices=self.edge_index, values=v, dtype=torch.long, device=self.edge_index.device)

    def merge_nodes(self, nodes_ndx):
        assert len(nodes_ndx) > 0 
        # merge attributes, preservice any side connection flags
        new_node_attr, _ = self.node_attr[nodes_ndx].max(dim=0)
        dadj = self.adjacency_matrix.to_dense()
        new_node_ndx = dadj.size(0)
        new_edge_row_mask = dadj[nodes_ndx, :].max(dim=0)[0].bool()
        new_edge_col_mask = dadj[:, nodes_ndx].max(dim=1)[0].bool()
        new_edge_row = torch.arange(len(new_edge_row_mask), device=self.device)[new_edge_row_mask]
        new_edge_col = torch.arange(len(new_edge_col_mask), device=self.device)[new_edge_col_mask]
        new_node_row_ndx = torch.full_like(new_edge_row, fill_value=new_node_ndx)
        new_node_col_ndx = torch.full_like(new_edge_col, fill_value=new_node_ndx)

        self.node_attr = torch.cat([self.node_attr, new_node_attr.unsqueeze(dim=0)])
        self.action_map = torch.nn.functional.pad(self.action_map, (0,0,0,1))
        self.edge_index = torch.cat([
            torch.stack([new_edge_row, new_node_row_ndx]),
            torch.stack([new_node_col_ndx, new_edge_col]),
            self.edge_index
        ], dim=1)

    def remove_nodes(self, nodes_ndx):
        if len(nodes_ndx) > 0:
            # remove orphaned edges
            dadj = self.adjacency_matrix.to_dense()
            new_size = dadj.size(0) - len(nodes_ndx)
            mask = torch.ones(dadj.size(), device=self.device).bool()
            mask[nodes_ndx, :] = False
            mask[:, nodes_ndx] = False
            adj = dadj[mask].view(new_size, new_size).to_sparse()
            self.edge_index = adj.indices()
            # remove node attributes
            mask = torch.ones_like(self.node_attr[:,0]).bool()
            nodes_ndx = nodes_ndx if isinstance(nodes_ndx, torch.Tensor) else torch.tensor(nodes_ndx, dtype=torch.long, device=self.device)
            mask.scatter_(0, nodes_ndx, False)
            self.node_attr = self.node_attr[mask]
            self.action_map = self.action_map[mask]

    def merge_groups(self):
        """
        merge connected groups of stones into single nodes, consolidating edges

        if untraversed stone nodes exist:
            pop next untraversed node
            find any neighbour stone nodes
            add them to the merge set

        merge node groups 

        """

        def get_group(node_ndx):
            player = self.node_attr[node_ndx, 0]
            to_do = set([node_ndx])
            done = set()

            while len(to_do) > 0:
                node_ndx = to_do.pop()
                done.add(node_ndx)
                edge_mask = self.edge_index[0, :] == node_ndx
                neighbour_ndx = self.edge_index[1, edge_mask]
                neighbour_ndx = neighbour_ndx[self.node_attr[neighbour_ndx, 0] == player]
                for n in neighbour_ndx:
                    if not n.item() in done:
                        to_do.add(n.item())
            return done

        #adj = self.adjacency_matrix
        stone_mask = (self.node_attr[:, 0] != 0).cpu()
        untraversed = set(np.arange(len(stone_mask))[stone_mask].tolist())
        old_node_ndxs = []
        
        while len(untraversed) > 0:
            node_ndx = untraversed.pop()
            group = get_group(node_ndx)
            if len(group) > 1:
                untraversed -= group
                old_node_ndxs += list(group)
                self.merge_nodes(torch.tensor(list(group), dtype=torch.long))
        
        self.remove_nodes(old_node_ndxs)

    def get_neighbourhood(self, node_ndx):
        """ get the indicies of the nodes that share an edge with the node at node_ndx
        """
        neighbourhood_mask = self.edge_index[0, :] == node_ndx
        return self.edge_index[1, neighbourhood_mask].unique()

        
class PlayerGraph(BoardGraph):
    def __init__(self, node_attr, edge_index, action_map, player):
        """
        self.d_edge_index - adjacency matrix (nodes, nodes) showing derived connections
        self.d_edge_connectivity - (edges in d_edge_index) values tensor for edge connectivity  i.e. how many stone must be placed to secure the edge
        self.d_carrier_index - adjacency matrix (edges in d_edge_index, nodes) showing derived carrier for each edge
        """
        super(PlayerGraph, self).__init__(node_attr, edge_index, action_map)
        self.player = player

    @classmethod
    def from_board_graph(cls, board_graph, player):
        assert player == -1 or player == 1

        non_player_node_ndx = (board_graph.node_attr[:, 0] == -player).nonzero().squeeze()
        player_graph = cls(board_graph.node_attr, board_graph.edge_index, board_graph.action_map, player)
        player_graph.remove_nodes(non_player_node_ndx)
        # in cannonical form player stone is always 1
        player_graph.node_attr[:,0] *= player
        #player_graph.reset_dgraph()

        return player_graph

    def get_node_attr(self, size, position_encoder=None, add_one_hot_node_id=False):
        a = self.node_attr.float()
        if add_one_hot_node_id:
            node_id = torch.eye(a.size(0), device=self.device)
            a = torch.cat([a, node_id], dim=1)
        if position_encoder is not None:
            # generate identifiers for the nodes based on a shuffled range of integers
            # this breaks up meaningless patterns in sequential node identifiers
            pos = position_encoder(torch.randperm(a.size(0), device=self.device))
            a = torch.cat([a, pos], dim=1)

        assert a.size(1) <= size, "node attributes too large"
        padding = size - a.size(1)
        a = torch.nn.functional.pad(a, (0,padding,0,0))

        return a

    def shortest_path(self):
        """ finds the shortest path between the two distingused terminal nodes
        measured in empty cells. 
        """
        def update_node_dist(node_ndx, dist):
            for n in node_ndx:
                if nd[n] > dist:
                    # increment distance if the node is an empty cell
                    ndist = dist
                    if self.node_attr[n, 0] == 0:
                        ndist += 1
                    nd[n] = ndist
                    nbr_ndx = self.get_neighbourhood(n)
                    update_node_dist(nbr_ndx, ndist)

        # indicies of terminal nodes
        tn1_ndx = self.node_attr[:, 1].bool().nonzero()[0,0]
        tn2_ndx = self.node_attr[:, 2].bool().nonzero()[0,0]
        # min distance for each node from t1
        nd = np.full((self.node_attr.size(0),), dtype=np.float, fill_value=math.inf)
        update_node_dist([tn1_ndx], 0)

        return nd[tn2_ndx]


    # def reset_dgraph(self):
    #     self.d_edge_index = torch.stack([
    #         torch.arange(self.node_attr.size(0)),
    #         torch.arange(self.node_attr.size(0))
    #     ])
    #     mask = (self.node_attr == 0)
    #     self.d_edge_connectivity = mask.long() 
    #     self.d_carrier_index = self.d_edge_index[:, mask].clone()

    # @property
    # def d_adjacency_matrix(self):
    #     return torch.sparse.LongTensor(self.d_edge_index, self.d_edge_connectivity + 1)

    # @property
    # def carrier_adjacency_matrix(self):
    #     v = torch.ones_like(self.d_carrier_index[0])
    #     return torch.sparse.LongTensor(self.d_carrier_index, v)

    #def edge_step(self):
        # for each stone group node
        #   for each connected empty cell node
        #       fore each connected node
        #           if not self loop, add derived edge to cell to d_edge list for group
        
        #   for each pair with matching endpoints


# from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=200, base_wave_length=5):
        super().__init__()
        assert((d_model % 2) == 0)
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (base_wave_length ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (base_wave_length ** ((2 * i)/d_model)))
                
        self.pe = pe #.unsqueeze(0)
        #self.register_buffer('pe', self.pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        #x = x * math.sqrt(self.d_model)
        device = x.device
        x = self.pe[x].to(device)

        return x

class NullPositionalEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        device = x.device
        x = torch.zeros(x.size(0), self.d_model)

        return x


# b = Board(torch.zeros(4,4).long())
# b.np_pieces[1,0] = -1
# b.np_pieces[1,1] = -1
# b.np_pieces[2,1] = -1
# b.np_pieces[2,2] = -1
# b.np_pieces[1,3] = -1
# b.np_pieces[0,2] = 1
# b.np_pieces[0,3] = 1
# b.np_pieces[3,2] = 1
# print(b.display_string, "\n\n", b.np_pieces)
# g = BoardGraph.graph_from_board(b)
# print("\n\nBefore merge", g)
# g.merge_groups()
# print("After merge", g)
# print(g.node_attr)
# print(g.edge_index)
# pg = PlayerGraph.from_board_graph(g, -1)
# print(b.display_string, "\n\n", b.np_pieces, "\n\n")
# print("H player", pg)