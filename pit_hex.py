import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

import Arena
from MCTS import MCTS
from hex.matrix_hex_game import MatrixHexGame
from hex.graph_hex_game import GraphHexGame
from hex.graph_hex_board import GraphHexBoard
from hex.NNet import NNetWrapper as NNet, FakeNNet, value_from_shortest_path
from hex.hex_players import RandomPlayer, HumanHexPlayer, UIPlayer, PureMCTSPlayer, GPureMCTSPlayer
from utils import dotdict, config_rec

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

flags.DEFINE_enum('board_type', 'hex',
    ['hex', 'vortex'],
    'hex: standard hex grid Hex board, '
    'vortex: random grap board played on a Voronoi diagram')
flags.DEFINE_enum('player2', 'nnet',
    ['nnet', 'human', 'random', 'MCTS'],
    'nnet: nnet vs nnet, '
    'human: play interactivly with a human, '
    'random: nnet vs random play agent, '
    'MCTS: nnet vs MCTS with random agent')
flags.DEFINE_boolean('verbose', False, 'show playout')
flags.DEFINE_boolean('graphic', False, 'show playout graphically')
flags.DEFINE_boolean('node_nums', False, 'show node numbers on UI')
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
    if FLAGS.board_type == 'hex':
        g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
        if FLAGS.graphic:
            raise Exception("graphic display not implemented for hex boards")
    elif FLAGS.board_type == 'vortex':
        board = GraphHexBoard.new_vortex_board(FLAGS.game_board_size)
        g = GraphHexGame(board)
        if FLAGS.verbose:
            raise Exception("ascii display not implemented for vortex boards")

    # nnet player 1
    n1 = NNet(g, net_type=FLAGS.p1_nnet)
    n1.load_checkpoint('./', FLAGS.cpu1_checkpoint)
    args1 = dotdict({'numMCTSSims': FLAGS.p1_MCTS_sims, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    player1 = get_action_func(mcts1)
    on_move_end = None
    on_game_end = None

    # player 2
    if FLAGS.player2 == 'human':
        if FLAGS.board_type == 'hex':
            player2 = HumanHexPlayer(g).play
        elif FLAGS.board_type == 'vortex':
            ui = UIPlayer(g, show_node_numbers=FLAGS.node_nums)
            on_move_end = ui.update
            player2 = ui.play
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
        if FLAGS.board_type == 'hex' or not FLAGS.graphic:
            mcts2 = PureMCTSPlayer(g, sims=FLAGS.p2_MCTS_sims)
        elif FLAGS.board_type == 'vortex':
            mcts2 = GPureMCTSPlayer(g, sims=FLAGS.p2_MCTS_sims, show_node_numbers=FLAGS.node_nums)
            on_move_end = mcts2.update
            on_game_end = mcts2.on_game_end
        player2 = mcts2.play

    if FLAGS.board_type == 'hex':
        arena = Arena.Arena(player1, player2, g, display=MatrixHexGame.display, display_move=g.display_move)
    elif FLAGS.board_type == 'vortex':
        arena = Arena.Arena(player1, player2, g, on_move_end=on_move_end, on_game_end=on_game_end)

    result = config_rec()
    result["p1_wins"], result["p2_wins"], _ = arena.playGames(FLAGS.num_games, verbose=FLAGS.verbose)
    print("Results:\n\tP1:{}\n\tP2:{}".format(result["p1_wins"], result["p2_wins"]))
    logging.info("result:" + str(result))


if __name__ == '__main__':
    app.run(main)
