import Arena
from MCTS import MCTS
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper as NNet
from hex.HexPlayers import RandomPlayer, HumanHexPlayer
from utils import dotdict


import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


flags.DEFINE_boolean('human_vs_cpu', False, 'play interactivly with a human')
flags.DEFINE_boolean('verbose', False, 'show playout')
flags.DEFINE_integer('num_games', 2, 'Number of games to play')
flags.DEFINE_string('cpu1_checkpoint', 'base_cnn/base_cnn_best.pth.tar', 'pretrained weights for computer player 1')
flags.DEFINE_string('cpu2_checkpoint', 'base_cnn/base_cnn_best.pth.tar', 'pretrained weights for computer player 2')
flags.DEFINE_integer('game_board_height', 7, 'overide default height')
flags.DEFINE_integer('game_board_width', 7, 'overide default width')

def main(_argv):
    g = HexGame(FLAGS.game_board_height, FLAGS.game_board_width)

    # all players
    rp = RandomPlayer(g).play
    hp = HumanHexPlayer(g).play
    
    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/hex/', FLAGS.cpu1_checkpoint)
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    if FLAGS.human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        n2.load_checkpoint('./pretrained_models/hex/', FLAGS.cpu2_checkpoint)
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=HexGame.display, display_move=g.display_move)

    print(arena.playGames(FLAGS.num_games, verbose=FLAGS.verbose))

if __name__ == '__main__':
    app.run(main)