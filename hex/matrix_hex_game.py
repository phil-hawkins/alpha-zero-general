import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .matrix_hex_board import MatrixHexBoard
from .board_graph import Board


class MatrixHexGame(Game):
    """
    Hex Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, np_pieces=None):
        Game.__init__(self)
        self._base_board = MatrixHexBoard(height, width, np_pieces)
        self.next_player = 1

    @property
    def board_size(self):
        assert self._base_board.width == self._base_board.height
        return self._base_board.width

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.height * self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""

        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        action = self.getCanonicalAction(action, player)
        b.add_stone(action, player)
        self.next_player = -player
        return b.np_pieces, self.next_player

    def getValidMoves(self, board, player=None):
        """Any empty cell is a valid move"""
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
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

    def getCanonicalAction(self, action, player):
        if player == -1:
            r, c = divmod(action, self._base_board.height)
            action = (c * self._base_board.width) + r
        return action

    def getCanonicalForm(self, board, player):
        if player == -1:
            # Flip player from 1 to -1
            return np.transpose(board * player)
        else:
            return board

    def getSymmetries(self, board, pi):
        """Board is symmetrical under 180 degree rotation"""
        return [(board, pi), (np.rot90(board, 2), pi[::-1])]

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        print(Board.np_display_string(board))

    def player_name(self, player_num):
        name = 'V' if player_num == 1 else 'H'
        return name

    def display_move(self, action, player):
        action = self.getCanonicalAction(action, player)
        r, c = divmod(action, self._base_board.width)
        print("\nPlayer {} placed stone at {}{}".format(self.player_name(player), chr(r+ord('a')), c))
