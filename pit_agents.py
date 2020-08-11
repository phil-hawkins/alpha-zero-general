import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

import Arena
from MCTS import MCTS
from hex.matrix_hex_game import MatrixHexGame
from hex.graph_hex_game import GraphHexGame
from hex.graph_hex_board import GraphHexBoard
from hex.NNet import NNetWrapper as NNet
from hex.hex_players import HumanHexPlayer, UIPlayer, PureMCTSPlayer, GPureMCTSPlayer, PureNNetAgent
from utils import dotdict, config_rec

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

flags.DEFINE_enum('board_type', 'hex',
    ['hex', 'vortex'],
    'hex: standard hex grid Hex board, '
    'vortex: random grap board played on a Voronoi diagram')
flags.DEFINE_enum('agent1', 'NN',
    ['NN', 'MCTSnet', 'MCTS'],
    'NN: neural network policy only - no MCTS, '
    'MCTSnet: alpha zero style MCTS + neural network, '
    'MCTS: pure Monty Carlo Tree Search agent with full rollouts and not neural network')
flags.DEFINE_enum('agent2', 'NN',
    ['NN', 'MCTSnet', 'human', 'MCTS'],
    'NN: neural network policy only - no MCTS, '
    'MCTSnet: alpha zero style MCTS + neural network, '
    'human: interactive UI for human player. Only one human player allowed, '
    'MCTS: pure Monty Carlo Tree Search agent with full rollouts and not neural network')
flags.DEFINE_enum('UI', 'no_UI',
    ['ascii', 'graphic', 'no_UI'],
    'Must be ascii or no_UI for Hex and graphic or no_UI for Vortex')
flags.DEFINE_integer('num_games', 2, 'Number of games to play')
flags.DEFINE_string('agent1_nn_checkpoint', 'temp/gat/strong_5x5_b.pth.tar', 'pretrained weights for agent 1 if using a neural network')
flags.DEFINE_string('agent2_nn_checkpoint', 'temp/gat/strong_5x5_b.pth.tar', 'pretrained weights for agent 1 if using a neural network')
flags.DEFINE_string('agent1_nnet', 'base_gat', 'neural net for p,v estimation for agent 1 if using a neural network')
flags.DEFINE_string('agent2_nnet', 'base_gat', 'neural net for p,v estimation for agent 2 if using a neural network')
flags.DEFINE_integer('agent1_MCTS_sims', 100, 'number of simulated steps taken by tree search for agent 1 if using MCTS')
flags.DEFINE_integer('agent2_MCTS_sims', 100, 'number of simulated steps taken by tree search for agent 2 if using MCTS')
flags.DEFINE_integer('game_board_size', 5, 'overide default size')
flags.DEFINE_boolean('node_nums', False, 'show node numbers on UI')


def get_action_func(search):
    def action_func(x, p):
        return np.argmax(search.getActionProb(x, temp=0))

    return action_func


def main(_argv):
    on_move_end = None
    on_game_end = None
    on_display = None
    on_display_move = None

    if FLAGS.board_type == 'hex':
        g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
        on_display = MatrixHexGame.display
        on_display_move = g.display_move
        if FLAGS.UI == 'graphic':
            raise RuntimeError("graphic display not implemented for hex boards")
    elif FLAGS.board_type == 'vortex':
        board = GraphHexBoard.new_vortex_board(FLAGS.game_board_size)
        g = GraphHexGame(board)
        if FLAGS.UI == 'ascii':
            raise RuntimeError("ascii display not implemented for vortex boards")

    agents = []
    agent = None
    for a in ['agent1', 'agent2']:
        if FLAGS[a].value == 'human':
            if FLAGS.UI == 'no_UI':
                raise RuntimeError("a UI is required for a human agent to play")

            if FLAGS.board_type == 'hex':
                agent = HumanHexPlayer(g).play
            elif FLAGS.board_type == 'vortex':
                ui = UIPlayer(g, show_node_numbers=FLAGS.node_nums)
                on_move_end = ui.update
                agent = ui.play
        elif FLAGS[a].value == 'NN':
            model = NNet(g, net_type=FLAGS[a+'_nnet'].value)
            model.load_checkpoint('./', FLAGS[a+'_nn_checkpoint'].value)
            agent = PureNNetAgent(g, model).play
        elif FLAGS[a].value == 'MCTSnet':
            model = NNet(g, net_type=FLAGS[a+'_nnet'].value)
            model.load_checkpoint('./', FLAGS[a+'_nn_checkpoint'].value)
            args = dotdict({'numMCTSSims': FLAGS[a+'_MCTS_sims'].value, 'cpuct': 1.0})
            mcts = MCTS(g, model, args)
            agent = get_action_func(mcts)
        elif FLAGS[a].value == 'MCTS':
            if FLAGS.board_type == 'vortex' \
                    and FLAGS.UI == 'graphic' \
                    and FLAGS.agent2 != 'human':
                mcts = GPureMCTSPlayer(g, sims=FLAGS[a+'_MCTS_sims'].value, show_node_numbers=FLAGS.node_nums)
                on_move_end = mcts.update
                on_game_end = mcts.on_game_end
            else:
                mcts = PureMCTSPlayer(g, sims=FLAGS[a+'_MCTS_sims'].value)
            agent = mcts.play

        assert agent is not None
        agents.append(agent)

    arena = Arena.Arena(
        agents[0],
        agents[1],
        g,
        display=on_display,
        display_move=on_display_move,
        on_move_end=on_move_end,
        on_game_end=on_game_end)

    result = config_rec()
    result["agent1_p1_wins"], result["agent1_p2_wins"], result["agent2_p1_wins"], result["agent2_p2_wins"], _ = arena.playGames(FLAGS.num_games, verbose=(FLAGS.UI == 'ascii'), p_order_results=True)
    print("Results:\n\tA1:{} ({},{})\n\tA2:{} ({},{})".format(
        result["agent1_p1_wins"] + result["agent1_p2_wins"],
        result["agent1_p1_wins"], result["agent1_p2_wins"],
        result["agent2_p1_wins"] + result["agent2_p2_wins"],
        result["agent2_p1_wins"], result["agent2_p2_wins"]))
    logging.info("result:" + str(result))


if __name__ == '__main__':
    app.run(main)
