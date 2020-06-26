import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game
from connect4.pytorch.NNet import NNetWrapper as NNet
from connect4.Connect4Players import RandomPlayer, HumanConnect4Player, OneStepLookaheadConnect4Player
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
flags.DEFINE_string('cpu1_checkpoint', 'best.pth.tar', 'pretrained weights for computer player 1')
flags.DEFINE_string('cpu2_checkpoint', 'best.pth.tar', 'pretrained weights for computer player 2')
flags.DEFINE_integer('game_board_height', None, 'overide default height')
flags.DEFINE_integer('game_board_width', None, 'overide default width')
flags.DEFINE_integer('game_win_length', None, 'overide default win_length')

def main(_argv):
    g = Connect4Game(FLAGS.game_board_height, FLAGS.game_board_width, FLAGS.game_win_length)

    # all players
    rp = RandomPlayer(g).play
    oslp = OneStepLookaheadConnect4Player(g).play
    hp = HumanConnect4Player(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./pretrained_models/connect4/', FLAGS.cpu1_checkpoint)
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    if FLAGS.human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        n2.load_checkpoint('./pretrained_models/connect4/', FLAGS.cpu2_checkpoint)
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=Connect4Game.display)

    print(arena.playGames(FLAGS.num_games, verbose=FLAGS.verbose))

if __name__ == '__main__':
    app.run(main)