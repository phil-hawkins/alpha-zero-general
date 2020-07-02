import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanHexPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        print('\nMoves (RC) for 1 (CR) for -1:')

        while True:
            move_row = int(input())
            move_col = int(input())
            move = move_row * board.shape[1] + move_col
            if move < len(valid_moves) and valid_moves[move]: 
                break
            else: 
                print('Invalid move')
        return move

