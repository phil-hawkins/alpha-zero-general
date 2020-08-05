import numpy as np
from time import sleep
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import trange
from copy import deepcopy

from .players import HORIZONTAL_PLAYER, VERTICAL_PLAYER
from min_mcts.monte_carlo_tree_search import PureMCTS
from min_mcts.hex_node import HexNode

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


class PureMCTSPlayer():
    def __init__(self, game, sims):
        self.game = game
        self.sims = sims
        self.reset()

    def reset(self):
        self.tree = PureMCTS()

    def on_game_end(self):
        self.reset()

    def play(self, board, cur_player):
        HexNode.base_board = deepcopy(board)
        root_node = HexNode.from_hex_board(board)
        for _ in range(self.sims):
            self.tree.do_rollout(root_node)
        node = self.tree.choose(root_node)
        return node.last_action

class UIPlayer():

    def __init__(self, game, show_node_numbers=False, pickable=True):
        self.game = game
        self.vor_patches = None
        self.selected_node_ndx = None
        self.cell_colours = np.stack([
            mcolors.to_rgba("red"),
            mcolors.to_rgba("linen"),
            mcolors.to_rgba("blue")
        ])
        self.show_node_numbers = show_node_numbers
        self.fig = self.show_board(pickable)
        self.fig.canvas.draw_idle()
        for _ in range(100):
            self.fig.canvas.flush_events()

    def set_cell_colours(self, cell_states):
        self.vor_patches.set_facecolor(self.cell_colours[cell_states + 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def play(self, board, cur_player):
        cell_states = board.node_attr[:, 0].long() * cur_player
        self.set_cell_colours(cell_states)
        self.selected_node_ndx = None
        valid_moves = board.get_valid_moves()

        while True:
            # wait for player move
            while self.selected_node_ndx is None:
                sleep(0.01)
                self.fig.canvas.flush_events()

            # check that selection is valid
            if valid_moves[self.selected_node_ndx]:
                break
            else:
                print("Invalid action, node {}".format(self.selected_node_ndx))

        return self.selected_node_ndx

    def on_pick_node(self, event):
        artist = event.artist
        if isinstance(artist, PatchCollection):
            self.selected_node_ndx = event.ind[0]
            print('class onpick node:', self.selected_node_ndx)

    def update(self, board, cur_player=1):
        cell_states = board.node_attr[:, 0].long() * cur_player
        self.set_cell_colours(cell_states)

    def show_board(self, pickable=True):
        plt.ion()
        base_board = self.game.getInitBoard()
        vor = Voronoi(base_board.tri.points)

        plt.rcParams['figure.figsize'] = [10, 10]
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Vortex')

        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.8, point_size=2)

        patches = []
        for region in base_board.vor_regions:
            patches.append(Polygon(region))
        self.vor_patches = PatchCollection(patches, match_original=True, alpha=0.4, picker=1)
        self.vor_patches.set_facecolor(base_board.cell_colours)
        ax.add_collection(self.vor_patches)

        # add some colours to the sides
        bounds = np.array([
            [-0.1, -0.1],
            [-0.1, 1.1], 
            [1.1, 1.1]
        ])
        ax.plot(bounds[0], bounds[1], color=self.cell_colours[HORIZONTAL_PLAYER+1], linewidth=10, alpha=0.5)
        ax.plot(bounds[2], bounds[1], color=self.cell_colours[HORIZONTAL_PLAYER+1], linewidth=10, alpha=0.5)
        ax.plot(bounds[1], bounds[0], color=self.cell_colours[VERTICAL_PLAYER+1], linewidth=10, alpha=0.5)
        ax.plot(bounds[1], bounds[2], color=self.cell_colours[VERTICAL_PLAYER+1], linewidth=10, alpha=0.5)
        
        if pickable:
            fig.canvas.mpl_connect('pick_event', self.on_pick_node)

        ax.triplot(base_board.tri.points[:, 0], base_board.tri.points[:, 1], base_board.tri.simplices)
        ax.plot(base_board.tri.points[:, 0], base_board.tri.points[:, 1], 'o')

        # node numbers
        if self.show_node_numbers:
            for i, p in enumerate(base_board.tri.points):
                ax.text(p[0], p[1], str(i), fontsize=12)

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
        return fig


class GPureMCTSPlayer(UIPlayer, PureMCTSPlayer):
    def __init__(self, game, sims, show_node_numbers=False):
        UIPlayer.__init__(self, game, show_node_numbers, pickable=False)
        self.sims = sims
        self.reset()
    
    def play(self, board, cur_player):
        cell_states = board.node_attr[:, 0].long() * cur_player
        self.set_cell_colours(cell_states)
        action = PureMCTSPlayer.play(self, board, cur_player)

        return action
