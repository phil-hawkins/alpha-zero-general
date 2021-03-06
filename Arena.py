import logging
from tqdm import tqdm
from time import sleep

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, display_move=None, on_move_end=None, on_game_end=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.display_move = display_move
        self.on_move_end = on_move_end
        self.on_game_end = on_game_end

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("\nTurn ", str(it), "Player ", self.game.player_name(curPlayer))
                self.display(board)

            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer), curPlayer)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            if verbose and self.display_move is not None:
                self.display_move(action, curPlayer)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            if self.on_move_end is not None:
                self.on_move_end(board)

        if self.on_move_end is not None:
            sleep(3)
        if verbose:
            assert self.display
            log.info("Game over: Turn {} Result {}".format(it, self.game.getGameEnded(board, 1)))
            self.display(board)
        if self.on_game_end:
            self.on_game_end()

        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False, p_order_results=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon_p1, oneWon_p2 = 0, 0
        twoWon_p1, twoWon_p2 = 0, 0
        draws = 0
        with tqdm(range(num), desc="Arena.playGames (1)") as t:
            for _ in t:
                gameResult = self.playGame(verbose=verbose)
                if gameResult == 1:
                    oneWon_p1 += 1
                elif gameResult == -1:
                    twoWon_p2 += 1
                else:
                    draws += 1
                t.set_postfix(p1=oneWon_p1, p2=twoWon_p2)

        self.player1, self.player2 = self.player2, self.player1

        with tqdm(range(num), desc="Arena.playGames (2)") as t:
            for _ in t:
                gameResult = self.playGame(verbose=verbose)
                if gameResult == -1:
                    oneWon_p2 += 1
                elif gameResult == 1:
                    twoWon_p1 += 1
                else:
                    draws += 1
                t.set_postfix(p1=twoWon_p1, p2=oneWon_p2)

        if p_order_results:
            return oneWon_p1, oneWon_p2, twoWon_p1, twoWon_p2, draws
        else:
            return oneWon_p1+oneWon_p2, twoWon_p1+twoWon_p2, draws