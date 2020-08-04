"""
A wrapper for the graph Hex board class for use with the minimal MCTS implementation
"""
import numpy as np
from .monte_carlo_tree_search import Node
from hex.graph_hex_board import GraphHexBoard
from hex.matrix_hex_board import MatrixHexBoard

class HexNode(Node):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """
    base_board = None

    def __init__(self, state, player, last_action=None):
        self.player = player
        self.state = state
        self.state.flags.writeable = False
        self.winner = None
        self._is_terminal = None
        self.last_action = last_action

    @classmethod
    def from_hex_board(cls, board, is_terminal=False, winner=None):
        if type(board).__module__ == np.__name__:
            n = cls(state=board.flatten(), player=1)
        else:
            n = cls(state=board.state_np, player=1)
        return n

    def _next_node(self, action):
        state = np.copy(self.state)
        state[action] = self.player

        return self.__class__(state, self.player*-1, last_action=action)

    def find_children(self):
        "All possible successors of this board state"
        children = set()
        board = self.__class__.base_board
        board.state_np = self.state
        actions = np.array(board.get_valid_moves())
        actions = actions.nonzero()[0]
        for a in actions:
            children.add(self._next_node(a.item()))

        return children

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        board = self.__class__.base_board
        board.state_np = self.state
        actions = np.array(board.get_valid_moves())
        actions = actions.nonzero()[0]
        a = np.random.choice(actions, 1).item()

        return self._next_node(a)

    def is_terminal(self):
        "Returns True if the node has no children"
        self.__class__.base_board.state_np = self.state
        win_state = self.__class__.base_board.get_win_state()
        self._is_terminal = win_state.is_ended
        if win_state.is_ended:
            self.winner = win_state.winner

        return win_state.is_ended

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        assert self._is_terminal
        return 1 if self.winner == self.player else 0

    def __hash__(self):
        return hash(self.state.tobytes())

    def __eq__(self, node2):
        return np.array_equal(self.state, node2.state)
