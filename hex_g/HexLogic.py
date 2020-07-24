from collections import namedtuple
import numpy as np
import torch
from random import randint
from hex.pytorch.board_graph import BoardGraph


WinState = namedtuple('WinState', ['is_ended', 'winner'])


class Board():
    """
    Hex Board played on an a graph.
    """

    def __init__(self, graph):
        """Set up initial board configuration."""
        self.max_diameter = max_diameter
        self.winner = None
        self.g = graph
        for i in range(len(self.g.nodes)):
            self.g.nodes[i]['state'] = 0

    def add_stone(self, action, player):
        assert self.g['state'] == 0
        self.g['state'] = player
    
    def get_valid_moves(self):
        "Any zero value is a valid move"
        valids = []
        for i in range(len(self.g.nodes)):
            append(self.g.nodes[i]['state'] == 0)
        return valids

    def neighbors(self, cell, board_state):
        """ get the 1-hop adjacent cells to a given cell
        @param cell: input cell
        @returns: tensor of adjacent cell indicies
        """
        n = self.nkernel + cell
        on_board_mask = (
            (n[:,0] >= 0) & \
            (n[:,0] < board_state.shape[0]) & \
            (n[:,1] >= 0) & \
            (n[:,1] < board_state.shape[1]) 
        )
        n = n[on_board_mask]

        return n

    def get_win_state(self):
        """ checks whether player -1 has made a left-right connection or 
        player -1 has made a top-bottom connection by first transposing the board
        """
        def check_left_right_connect(board_state, player):
            # add a row of active player stones to the left side
            board_state = np.concatenate((
                np.ones((board_state.shape[0], 1), dtype=int) * player, 
                board_state
            ), axis=1)

            # see if we can connect the top left stone to the right side
            visited = []
            connected = [(0,0)]
            imax = -1

            while connected:
                stone = connected.pop()
                imax = max(imax, stone[1])
                # check whether there is a side to side connection
                if imax == board_state.shape[1] - 1:
                    return True
                else:
                    for n in self.neighbors(stone, board_state):
                        n = tuple(n)
                        if n not in visited:
                            if board_state[n] == player:
                                connected.append(n)
                visited.append(stone)

            return False        
        
        board_state = self.np_pieces.copy()
        for player in [-1, 1]:
            if check_left_right_connect(board_state, player):
                return WinState(True, player)
            board_state = np.transpose(board_state)

        return WinState(False, None)


    @classmethod
    def np_display_string(cls, np_pieces):
        b = Board(np_pieces.shape[0], np_pieces.shape[1], np_pieces)
        ds = b.display_string

        return ds

    @property
    def display_string(self):
        return DisplayBoard(torch.tensor(self.np_pieces)).display_string
        # board_str = "   "
        # for c in range(self.np_pieces.shape[1]):
        #     board_str += "{}   ".format(c)
        # board_str += "\n  "
        # for c in range(self.np_pieces.shape[1]):
        #     board_str += "----"
        # for r in range(self.np_pieces.shape[0]):
        #     board_str += "\n{}{} ` ".format(r*"  ", r)
        #     for c in range(self.np_pieces.shape[1]):
        #         board_str += " {: } ".format(self.np_pieces[r,c])
        #     board_str += "`"
        # board_str += "\n    {}".format(r*"  ")
        # for c in range(self.np_pieces.shape[1]):
        #     board_str += "----"

        # return board_str

    def __str__(self):
        return str(self.np_pieces)
