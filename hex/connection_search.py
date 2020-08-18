# import numpy as np
# import torch
# from collections import namedtuple

# CNode = namedtuple('ConnectionNode', ['left_connected', 'right_connected'])
# EPS = 1e-8

# class ConnectionSearch():

#     def __init__(self, canonicalBoard, c_net, args):
#         # the connectability network determines how close cells are in terms of effort to connect
#         self.args = args
#         self.c_net = c_net
#         self.board_size = canonicalBoard.size(0)
#         # add a row of active player stones to the left and right sides of the board
#         self.canonicalBoard = torch.nn.functional.pad(canonicalBoard, (1, 1), "constant", 1)
#         # the connectability graph maps visited cells to the cannonical board
#         self.c_nodes = {}
#         #self.c_edges = {}
#         self.edge_visits = {}
#         self.node_visits = {}
#         self.edge_resistence = {}
#         self.connected_nodes = set()
#         self.cs = {}       # the connection strength predictions for visited cells

#     def get_player_connections(self):
#         tr_cell, br_cell = (0,0), (self.board_size-1,self.board_size+1)
#         self.c_nodes = {
#             tr_cell : CNode(left_connected=True, right_connected=False),
#             br_cell : CNode(left_connected=False, right_connected=True)
#         }
#         for i in range(self.args.search_count // 2):
#             for cell in [tr_cell, br_cell]:
#                 self.search(cell)

#         self.propagate_connected_nodes()

#     def propagate_connected_nodes(self):
#         cn = self.connected_nodes
#         node_edges = {}
#         for edge_key in self.edge_visits.keys():
#             for [node in edge_key]:
#                 if node in node_edges:
#                     node_edges[node].append(edge_key)
#                 else:
#                     node_edges[node] = [edge_key]

#         while(True):
#             node = cn.pop()
#             self.connected_nodes.add(node)
#             for edge_key in node_edges[node]:
#                 for [enode in edge_key]:
#                     if (enode != node):
#                         if not node in self.connected_nodes:
#                             cn.add(enode)
#             if len(cn) == 0:
#                 break


#     def get_edge_key(self, from_cell, to_cell):
#         edge_key = (from_cell, to_cell)
#         if (edge_key not in self.edge_visits):
#             if (to_cell, from_cell) in self.edge_visits:
#                 edge_key = (to_cell, from_cell)
#             else:
#                 edge_key = None

#         return edge_key

#     def get_edge_visit_count(self, from_cell, to_cell):
#         edge_key = self.get_edge_key(from_cell, to_cell)
#         if edge_key is None:
#             return 0
#         else:
#             return self.edge_visits[edge_key]

#     def search(self, cell):
#         if cell not in self.cs:
#             # leaf node, predict resistence estimates for all cells within sight of the current search cell
#             self.cs[cell] = self.c_net(self.canonicalBoard, board_cell)
#             self.node_visits[cell] = 0
#         else:
#             # get resistence estimates
#             cond = self.cs[cell]
#             # pick the cell with the best u 
#             # where u is a function of resistence and the visit count of the edge
#             cur_best = -float('inf')
#             best_cell = None

#             for r in range(self.board_size):
#                 for c in range(self.board_size + 2):
#                     to_cell = (r, c)
#                     ev = self.get_edge_visit_count(cell, to_cell)
#                     nv = self.node_visits[cell]
#                     u = cond[to_cell] + self.args.exp_c * math.sqrt(nv + EPS) / (1 + ev)

#                     if u > cur_best:
#                         cur_best = u
#                         best_cell = to_cell
#             to_cell = best_cell

#             # add or update edge
#             edge_key = self.get_edge_key(cell, to_cell)
#             if edge_key is None:
#                 self.c_edges[(cell, to_cell)] = CEdge(resistence=cond[to_cell], visit_count=1)
#                 if to_cell not in self.c_nodes:
#                     self.c_nodes[to_cell] = self.c_nodes[cell]
#             else:
#                 self.edge_visits[edge_key] += 1
#                 # TODO: if the edge exists in the other direction and average the resistence estimate

#             self.node_visits[cell] += 1

#             # continue searching until a side to side path is found
#             if (self.c_nodes[cell].left_connected and self.c_nodes[to_cell].right_connected) or
#                 (self.c_nodes[to_cell].left_connected and self.c_nodes[cell].right_connected):
#                 for node in edge_key:
#                     self.connected_nodes.add(node)
#             else:
#                 self.search(to_cell)

