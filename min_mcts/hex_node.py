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

    def __init__(self, state, next_player, last_action=None):
        self.next_player = next_player
        self.state = state
        self.state.flags.writeable = False
        self.last_action = last_action
        
        # get win state
        self.__class__.base_board.state_np = self.state
        win_state = self.__class__.base_board.get_win_state()
        self._is_terminal = win_state.is_ended
        self.winner = win_state.winner

    @classmethod
    def from_hex_board(cls, board, is_terminal=False, winner=None):
        if type(board).__module__ == np.__name__:
            n = cls(state=board.flatten(), next_player=1)
        else:
            n = cls(state=board.state_np, next_player=1)
        return n

    @property
    def last_player(self):
        return -self.next_player

    def _next_node(self, action):
        state = np.copy(self.state)
        state[action] = self.next_player

        return self.__class__(state, self.next_player*-1, last_action=action)

    def find_children(self):
        "All possible successors of this board state"
        children = set()
        if not self._is_terminal:
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
        return self._is_terminal

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"

        # modified from original to fit with alpha zero general cannonical board where the board is 
        # always arranged such that the current player is 1 and the opponent -1        
        #
        # if not self._is_terminal:
        #     raise RuntimeError(f"reward called on nonterminal board")
        # if self.winner == self.next_player:
        #     # It's your turn and you've already won. Should be impossible.
        #     raise RuntimeError(f"reward called on unreachable board")
        # if self.winner == -self.next_player:
        #     return 0  # Your opponent has just won. Bad.
        # raise RuntimeError(f"board has unknown winner")

        if not self._is_terminal:
            raise RuntimeError(f"reward called on nonterminal board")
        assert self.winner == self.last_player

        return 1


    def __hash__(self):
        return hash(self.state.tobytes())

    def __eq__(self, node2):
        return np.array_equal(self.state, node2.state)
