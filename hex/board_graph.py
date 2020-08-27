import numpy as np
import torch
import math

from .graph_hex_board import GraphHexBoard
from .players import HORIZONTAL_PLAYER, VERTICAL_PLAYER, EMPTY_CELL


class Board():
    def __init__(self, np_pieces):
        assert np_pieces.shape[0] == np_pieces.shape[1]
        self.np_pieces = np_pieces

    @property
    def display_string(self):
        display_chars = {
            EMPTY_CELL: ".",
            HORIZONTAL_PLAYER: "H",
            VERTICAL_PLAYER: "V"
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
                board_str += "  {} ".format(display_chars[self.np_pieces[r, c].item()])
            board_str += "`"
        board_str += "\n    {}".format(r*"  ")
        for c in range(self.np_pieces.shape[1]):
            board_str += "----"

        return board_str

    @property
    def size(self):
        return self.np_pieces.shape[0]

    @property
    def cell_count(self):
        return self.np_pieces.numel()

    @classmethod
    def np_display_string(cls, np_pieces):
        return cls(np_pieces).display_string


class BoardGraph():
    def __init__(self, node_attr, edge_index, action_map):
        self.node_attr = node_attr
        self.edge_index = edge_index
        self.action_map = action_map

    def __str__(self):
        return "Node Attributes:\n{}\n\nAdjacency Matrix:\n{}\n\nBoard Map: {}\n".format(self.node_attr, self.adjacency_matrix.to_dense(), self.action_map)

    @property
    def device(self):
        d = self.node_attr.device
        assert self.edge_index.device == d
        assert self.action_map.device == d

        return d

    @device.setter
    def device(self, d):
        self.node_attr = self.node_attr.to(device=d)
        self.edge_index = self.edge_index.to(device=d)
        self.action_map = self.action_map.to(device=d)

    @classmethod
    def from_graph_board(cls, board):
        assert isinstance(board, GraphHexBoard)

        action_map = torch.stack([
            torch.arange(board.node_attr.size(0), device=board.node_attr.device),   # board_cell
            (board.node_attr[:, 0] == 0).long()                                     # valid_action (1/0)
        ], dim=1)

        return cls(board.node_attr, board.edge_index, action_map)

    @classmethod
    def from_matrix_board(cls, board):
        """ create directed edge index for a standard hex board

        returns:
            graph object
        """
        board = Board(np_pieces=board)

        device = board.np_pieces.device
        k = torch.tensor([
            [-1, -1,  0,  1,  1,  0],
            [0,  1,  1,  0, -1, -1]
        ], device=device).unsqueeze(dim=1).expand(2, board.cell_count, 6).reshape(2, -1)
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
        edge_index = torch.cat([edge_index, side_edge_index, side_edge_index[[1, 0]]], dim=1)
        board_node_attr = torch.zeros((board.np_pieces.numel(), 3), dtype=torch.long, device=device)
        board_node_attr[:, 0] = board.np_pieces.flatten()
        side_node_attr = torch.tensor([
            [HORIZONTAL_PLAYER, 1, 0],     # player (H) side 1
            [HORIZONTAL_PLAYER, 0, 1],     # player (H) side 2
            [VERTICAL_PLAYER, 1, 0],      # player (V) side 1
            [VERTICAL_PLAYER, 0, 1]       # player (V) side 2
        ], dtype=torch.long, device=device)

        node_attr = torch.cat([board_node_attr, side_node_attr])
        action_map = torch.stack([
            torch.arange(node_attr.size(0), device=device),  # board_cell
            (node_attr[:, 0] == EMPTY_CELL).long()           # valid_action (1/0)
        ], dim=1)

        return cls(node_attr, edge_index, action_map)

    @property
    def adjacency_matrix(self):
        v = torch.ones_like(self.edge_index[0])
        n = self.node_attr.size(0)
        return torch.sparse_coo_tensor(indices=self.edge_index, values=v, dtype=torch.long, size=(n, n), device=self.edge_index.device)

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
        self.action_map = torch.nn.functional.pad(self.action_map, (0, 0, 0, 1))
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
            mask = torch.ones_like(self.node_attr[:, 0]).bool()
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

    def get_node_attr(self, size, id_encoder=None, add_one_hot_node_id=False):
        a = self.node_attr.float()
        if add_one_hot_node_id:
            node_id = torch.eye(a.size(0), device=self.device)
            a = torch.cat([a, node_id], dim=1)
        if id_encoder is not None:
            # generate identifiers for the nodes based on a shuffled range of integers
            # this breaks up meaningless patterns in sequential node identifiers
            ids = id_encoder(torch.randperm(a.size(0), device=self.device))
            a = torch.cat([a, ids], dim=1)

        assert a.size(1) <= size, "node attributes too large"
        padding = size - a.size(1)
        a = torch.nn.functional.pad(a, (0, padding, 0, 0))

        return a

    def state_to_planes(self):
        """ split attributes into planes for (player V, player H, empty)
        """
        cell_state = self.node_attr[:, 0].unsqueeze(dim=1)
        player_v = (cell_state == VERTICAL_PLAYER).float()
        player_h = (cell_state == HORIZONTAL_PLAYER).float()
        empty = (cell_state == EMPTY_CELL).float()
        self.node_attr = torch.cat([
            player_v,
            player_h,
            empty,
            self.node_attr[:, 1:].float()], dim=1)


class PlayerGraph(BoardGraph):
    def __init__(self, node_attr, edge_index, action_map, player, edge_index_2bridge=None):
        """
        self.d_edge_index - adjacency matrix (nodes, nodes) showing derived connections
        self.d_edge_connectivity - (edges in d_edge_index) values tensor for edge connectivity  i.e. how many stone must be placed to secure the edge
        self.d_carrier_index - adjacency matrix (edges in d_edge_index, nodes) showing derived carrier for each edge
        """
        super(PlayerGraph, self).__init__(node_attr, edge_index, action_map)
        self.player = player
        self.edge_index_2bridge = edge_index_2bridge

    def calc_2bridge_edge_index(self):
        A = self.adjacency_matrix.to_dense().float()
        A = (A.matmul(A) - A - 1).relu()
        # remove self loops
        A.fill_diagonal_(0.)
        self.edge_index_2bridge = A.nonzero().T

    @classmethod
    def from_board_graph(cls, board_graph, player):
        assert player == -1 or player == 1

        non_player_node_ndx = (board_graph.node_attr[:, 0] == -player).nonzero().squeeze()
        player_graph = cls(board_graph.node_attr, board_graph.edge_index, board_graph.action_map, player)
        player_graph.remove_nodes(non_player_node_ndx)
        player_graph.calc_2bridge_edge_index()
        # in cannonical form player stone is always 1
        player_graph.node_attr[:, 0] *= player
        # player_graph.reset_dgraph()

        return player_graph

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
        tn1_ndx = self.node_attr[:, 1].bool().nonzero()[0, 0]
        tn2_ndx = self.node_attr[:, 2].bool().nonzero()[0, 0]
        # min distance for each node from t1
        nd = np.full((self.node_attr.size(0),), dtype=np.float, fill_value=math.inf)
        update_node_dist([tn1_ndx], 0)

        return nd[tn2_ndx]


def batch_to_net(x, args, device):
    """ converts a batch of boards to input features suitable for the graph net

    input x is one of:
        - np.array (row, col)
        - np.array (batch, row, col)
        - GraphHexBoard object
        - list of GraphHexBoard objects

    returns:
        each of the following is a 2 element list with the first item for the current player
        and the second item for the opposing player:

        - edge_index : coo index for "super-graph" adjacency matrix (2=from/to, edges)
                this is a union of all graphs in the batch
        - node_attr : node embedding as column vectors (node, attributes)
        - batch: (nodes, features (0-2 below))
                0 : mapping from nodes to input graphs, as per torch geo batch
                1 : mask for nodes which are valid actions i.e. empty cells
                2 : map from node index to action index in original input

    """
    # marshall the input into a batch tensor or a list of graph objects
    if type(x).__module__ == np.__name__ or isinstance(x, torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x.astype(np.float64)).to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)

        board_to_graph = BoardGraph.from_matrix_board
    elif isinstance(x, GraphHexBoard):
        x = [x]
        board_to_graph = BoardGraph.from_graph_board
    elif type(x) == list:
        board_to_graph = BoardGraph.from_graph_board
    else:
        raise Exception("Unsupported batch type for convertion to graph net input")

    # split out the player graphs and build graph net input features from the boards
    player_graph = [None, None]
    edge_index = [[], [], [], []]
    node_attr = [[], []]
    node_ndx_start = [0, 0]
    batch = [[], []]
    for bi, board in enumerate(x):
        bg = board_to_graph(board)
        bg.device = device
        bg.merge_groups()
        for i, player in enumerate([-1, 1]):
            player_graph[i] = PlayerGraph.from_board_graph(bg, player)
            edge_index[i].append(player_graph[i].edge_index + node_ndx_start[i])
            edge_index[i+2].append(player_graph[i].edge_index_2bridge + node_ndx_start[i])
            a = player_graph[i].get_node_attr(size=args.num_channels, id_encoder=args.id_encoder)
            node_attr[i].append(a)
            node_ndx_start[i] += a.size(0)
            g = torch.cat((
                torch.full((a.size(0), 1), dtype=torch.long, fill_value=bi, device=device),
                player_graph[i].action_map
            ), dim=1)
            batch[i].append(g)

    for i in range(2):
        # one-hop edge index
        edge_index[i] = torch.cat(edge_index[i], dim=1)
        # 2 bridge edge index
        edge_index[i+2] = torch.cat(edge_index[i+2], dim=1)
        node_attr[i] = torch.cat(node_attr[i], dim=0)
        batch[i] = torch.cat(batch[i], dim=0)

    return edge_index, node_attr, batch


def batch_to_1trunk_net(x, args, device):
    """ converts a batch of boards to input features suitable for the one trunk graph net

    input x is one of:
        - np.array (row, col)
        - np.array (batch, row, col)
        - GraphHexBoard object
        - list of GraphHexBoard objects

    returns:
        - edge_index : coo index for "super-graph" adjacency matrix (2=from/to, edges)
                this is a union of all graphs in the batch
        - node_attr : node embedding as column vectors (node, attributes)
        - batch: (nodes, features (0-2 below))
                0 : mapping from nodes to input graphs, as per torch geo batch
                1 : mask for nodes which are valid actions i.e. empty cells
                2 : map from node index to action index in original input

    """
    # marshall the input into a batch tensor or a list of graph objects
    if type(x).__module__ == np.__name__ or isinstance(x, torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x.astype(np.float64)).to(device)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)

        board_to_graph = BoardGraph.from_matrix_board
    elif isinstance(x, GraphHexBoard):
        x = [x]
        board_to_graph = BoardGraph.from_graph_board
    elif type(x) == list:
        board_to_graph = BoardGraph.from_graph_board
    else:
        raise Exception("Unsupported batch type for convertion to graph net input")

    # build graph net input features from the boards
    edge_index, node_attr = [], []
    node_ndx_start = 0
    batch = []
    for bi, board in enumerate(x):
        bg = board_to_graph(board)
        bg.device = device
        bg.merge_groups()
        edge_index.append(bg.edge_index + node_ndx_start)
        # split attrributes into planes for (player V, player H, empty)
        bg.state_to_planes()
        a = bg.get_node_attr(size=args.num_channels, id_encoder=args.id_encoder)
        node_attr.append(a)
        node_ndx_start += a.size(0)
        g = torch.cat((
            torch.full((a.size(0), 1), dtype=torch.long, fill_value=bi, device=device),
            bg.action_map
        ), dim=1)
        batch.append(g)

    edge_index = torch.cat(edge_index, dim=1)
    node_attr = torch.cat(node_attr, dim=0)
    batch = torch.cat(batch, dim=0)

    return edge_index, node_attr, batch


# from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class IdentifierEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=200, base_wave_length=5):
        super().__init__()
        assert (d_model % 2) == 0, "identifier embedding must have an even number of dimentions"
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (base_wave_length ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (base_wave_length ** ((2 * i)/d_model)))

        self.pe = pe  # .unsqueeze(0)
        # self.register_buffer('pe', self.pe)

    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        device = x.device
        x = self.pe[x].to(device)

        return x


class ZeroIdentifierEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        device = x.device
        x = torch.zeros(x.size(0), self.d_model, device=device)

        return x


class RandomIdentifierEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        device = x.device
        x = torch.rand(x.size(0), self.d_model, device=device)

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
