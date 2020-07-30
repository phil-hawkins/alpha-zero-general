import sys
import numpy as np
from copy import deepcopy

sys.path.append('..')
from Game import Game
from .graph_hex_board import GraphHexBoard


class GraphHexGame(Game):
    """
    Hex on graph game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, board):
        Game.__init__(self)
        self._base_board = board
        self.next_player = 1

    def getInitBoard(self):
        return self._base_board

    def getActionSize(self):
        return self._base_board.action_size

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        board = deepcopy(board)
        board.add_stone(action, player)
        self.next_player = -player
        return board, self.next_player

    def getValidMoves(self, board, player):
        """Any empty cell is a valid move"""
        return board.get_valid_moves()

    def getGameEnded(self, board, player):
        winstate = board.get_win_state()
        if winstate.is_ended:
            if winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        if player == -1:
            # Flip player from 1 to -1
            return board.compliment()
        else:
            return board

    def getSymmetries(self, board, pi):
        """ No symmetry """
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.state_representation

