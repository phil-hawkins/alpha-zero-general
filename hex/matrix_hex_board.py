from collections import namedtuple
import numpy as np
import torch

WinState = namedtuple('WinState', ['is_ended', 'winner'])


class MatrixHexBoard():
    """
    Hex Board.
    """
    nkernel = np.array([
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 0],
        [1, -1],
        [0, -1]
    ])

    def __init__(self, height=None, width=None, np_pieces=None):
        "Set up initial board configuration."
        self.height = height or np_pieces.size(0)
        self.width = width or np_pieces.size(1)
        self.winner = None

        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width], dtype=np.int)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

    def add_stone(self, action, player):
        r, c = divmod(action, self.width)
        self.add_stone_rc(r, c, player)
    
    def add_stone_rc(self, row, column, player):
        "Create copy of board containing new stone."
        if self.np_pieces[row, column] != 0:
            raise ValueError("Can't play ({},{}) on board \n{}".format(row, column, self.display_string))

        self.np_pieces[row, column] = player

    def get_valid_moves(self):
        "Any zero value is a valid move"
        return self.np_pieces.reshape(-1) == 0

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

    def with_np_pieces(self, np_pieces):
        """Create copy of board with specified np_pieces."""
        if np_pieces is None:
            np_pieces = self.np_pieces
        return self.__class__(self.height, self.width, np_pieces)

    def __str__(self):
        return str(self.np_pieces)
