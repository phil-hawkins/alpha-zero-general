import unittest
import os,sys; sys.path.insert(0, os.path.abspath('.'))
import numpy as np

from hex.HexLogic import Board

class Test_Hex(unittest.TestCase):

    def test_board(self):
        b = Board(height=7, width=5)
        b.add_stone_rc(0, 0, -1)
        b.add_stone_rc(5, 2, 1)

        print(b)

    def test_winstate(self):
        # set up almost connected board
        b = Board(height=5, width=5)
        b.add_stone_rc(1, 0, -1)
        b.add_stone_rc(1, 1, -1)
        b.add_stone_rc(2, 2, -1)
        b.add_stone_rc(2, 3, -1)
        b.add_stone_rc(1, 4, -1)

        ws = b.get_win_state()
        self.assertFalse(ws.is_ended)
        
        # make the connection for player -1
        b.add_stone_rc(1, 2, -1)
        ws = b.get_win_state()
        self.assertTrue(ws.is_ended)
        self.assertEqual(ws.winner, -1)

        # transform board to a player 1 win
        b.np_pieces = np.transpose(b.np_pieces) * -1
        ws = b.get_win_state()
        self.assertTrue(ws.is_ended)
        self.assertEqual(ws.winner, 1)

        # break the connection
        b.np_pieces[0,1] = -1
        ws = b.get_win_state()
        self.assertFalse(ws.is_ended)


