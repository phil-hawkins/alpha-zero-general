import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, cur_player):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanHexPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, cur_player):
        valid_moves = self.game.getValidMoves(board, 1)
        print('\nMoves (RC):')

        while True:
            move_str = input()
            move_row = ord(move_str[0])-ord('a')
            move_col = int(move_str[1:])
            if cur_player == 1:
                move = move_row * board.shape[1] + move_col
            else:
                move = move_col * board.shape[1] + move_row
            if move < len(valid_moves) and valid_moves[move]: 
                break
            else: 
                print('Invalid move')
        return move

