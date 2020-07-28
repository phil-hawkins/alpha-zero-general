import Arena
from MCTS import MCTS
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper as NNet, FakeNNet, value_from_shortest_path
from hex.HexPlayers import RandomPlayer, HumanHexPlayer
from utils import dotdict


import numpy as np
from absl import app, flags
from absl.flags import FLAGS

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


flags.DEFINE_enum('player2', 'nnet',
    ['nnet', 'human', 'random', 'MCTS'],
    'nnet: nnet vs nnet, '
    'human: play interactivly with a human, '
    'random: nnet vs random play agent, '
    'MCTS: nnet vs MCTS with random agent')
flags.DEFINE_boolean('verbose', False, 'show playout')
flags.DEFINE_integer('num_games', 2, 'Number of games to play')
flags.DEFINE_string('cpu1_checkpoint', 'temp/gat/strong_5x5_b.pth.tar', 'pretrained weights for computer player 1')
flags.DEFINE_string('cpu2_checkpoint', 'temp/gat/strong_5x5_b.pth.tar', 'pretrained weights for computer player 2')
flags.DEFINE_integer('p1_MCTS_sims', 100, 'number of simulated steps taken by tree search for player 1')
flags.DEFINE_integer('p2_MCTS_sims', 100, 'number of simulated steps taken by tree search for player 2 if usincg MCTS')
flags.DEFINE_integer('game_board_size', 5, 'overide default size')
flags.DEFINE_string('p1_nnet', 'base_gat', 'neural net for p,v estimation for player 1')
flags.DEFINE_string('p2_nnet', 'base_gat', 'neural net for p,v estimation for player 2')


def get_action_func(search):
    def action_func(x, p):
        return np.argmax(search.getActionProb(x, temp=0))

    return action_func


def main(_argv):
    g = HexGame(FLAGS.game_board_size, FLAGS.game_board_size)

    # nnet player 1
    n1 = NNet(g, net_type=FLAGS.p1_nnet)
    n1.load_checkpoint('./', FLAGS.cpu1_checkpoint)
    args1 = dotdict({'numMCTSSims': FLAGS.p1_MCTS_sims, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x, p: np.argmax(mcts1.getActionProb(x, temp=0))

    # player 2
    if FLAGS.player2 == 'human':
        player2 = HumanHexPlayer(g).play
    elif FLAGS.player2 == 'random':
        player2 = RandomPlayer(g).play
    elif FLAGS.player2 == 'nnet':
        n2 = NNet(g, net_type=FLAGS.p2_nnet)
        n2.load_checkpoint('./', FLAGS.cpu2_checkpoint)
        args2 = dotdict({'numMCTSSims': FLAGS.p2_MCTS_sims, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        #player2 = lambda x, p: np.argmax(mcts2.getActionProb(x, temp=0))
        player2 = get_action_func(mcts2)
    elif FLAGS.player2 == 'MCTS':
        n2 = FakeNNet(g, value_function=value_from_shortest_path)
        args2 = dotdict({'numMCTSSims': FLAGS.p2_MCTS_sims, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        # player2 = lambda x, p: np.argmax(mcts2.getActionProb(x, temp=0))
        player2 = get_action_func(mcts2)

    arena = Arena.Arena(n1p, player2, g, display=HexGame.display, display_move=g.display_move)
    print(arena.playGames(FLAGS.num_games, verbose=FLAGS.verbose))


if __name__ == '__main__':
    app.run(main)
