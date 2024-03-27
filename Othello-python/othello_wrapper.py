from constants import N, WHITE, BLACK, NONE, BLOCKING
import functools
import numpy as np
import othello
import random
import tensorflow as tf

class OthelloGame:
    def __str__(self):
        s = []
        char_dict = {WHITE: "o", BLACK: "x", NONE: ".", BLOCKING: "-"}
        s.append("WHITE TURN" if self.get_player_turn() == WHITE else "BLACK TURN")
        for y in range(N-1, -1, -1):
            line = []
            for x in range(N):
                line.append(" ")
                line.append(char_dict[self.game.board[y][x]])
                line.append(" ")
            s.append("".join(line))
        return "\n".join(s)

    def __init__(self):
        self.game = othello.new_OthelloGame()
        self._refresh_legal_moves_cache()

    def reset(self):
        self.game.reset()
        self._refresh_legal_moves_cache()

    def get_moves(self) -> tuple:
        return (self.game.whiteMoves, self.game.blackMoves)

    def get_moves_2(self) -> list:
        if self.get_player_turn() == WHITE:
            return self.game.whiteMoves
        else:
            return self.game.blackMoves

    def make_move(self, whiteMove: othello.Move, blackMove: othello.Move) -> None:
        self.game.do_move(whiteMove, blackMove)
        self._refresh_legal_moves_cache()

    def make_move_2(self, move: othello.Move) -> None:
        if self.get_player_turn() == WHITE:
            self.game.do_move(move, othello.get_null_move())
        else:
            self.game.do_move(othello.get_null_move(), move)
        self._refresh_legal_moves_cache()

    def board_to_tensor(self, colour, ignore_game_end=False) -> tf.Tensor:
        tensor = np.zeros((1, N, N, 1), dtype=np.float32)
        if not self.game_has_ended() or ignore_game_end:
            for y in range(N):
                for x in range(N):
                    colour_at_square = self.game.board[x][y]
                    if colour_at_square == colour:
                        tensor[0, y, x, 0] = 1
                    elif colour_at_square == NONE:
                        pass
                    elif colour_at_square == BLOCKING:
                        pass
                    else:
                        tensor[0, y, x, 0] = -1
                    #if (x, y) in self.legal_moves_cache:
                    #    tensor[0, x, y, 2] = 1
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        return tensor

    def board_to_legal_moves(self):
        tensor = np.zeros((1, N * N - 4), dtype=np.float32)
        for i in range(N * N - 4):
            if self.number_is_legal(i):
                tensor[0, i] = 1.0
        return tf.convert_to_tensor(tensor, dtype=tf.float32)

    def _refresh_legal_moves_cache(self):
        self.legal_moves_cache = set()
        moves = self.get_moves_2()
        for _move in moves:
            self.legal_moves_cache.add((_move.x, _move.y))
        return self.legal_moves_cache

    def get_legal_numbers(self):
        return [i for i in range(N * N - 4) if self.number_is_legal(i)]

    def get_legal_moves(self):
        return sorted(list(self.legal_moves_cache))

    def number_to_move(self, x, y) -> othello.Move:
        move = othello.get_move(x, y)
        return move
    
    def set_square(self, x, y, colour):
        self.game.set_square(x, y, colour)
        self._refresh_legal_moves_cache()

    def get_player_turn(self):
        return self.game.colour_turn

    def number_is_legal(self, i) -> bool:
        (x, y) = self.number_to_cord(i)
        move = self.number_to_move(x, y)
        is_legal = (move.x, move.y) in self.legal_moves_cache
        return is_legal

    def get_winner(self):
        return self.game.winner

    def game_has_ended(self) -> bool:
        return self.game.game_has_ended()

    @functools.lru_cache(maxsize=100)
    def number_to_cord(self, i):
        for (x, y) in [(x, y) for x in range(N) for y in range(N)]:
            _x = x
            _y = y
            if not (x, y) in {(3, 3), (3, 4), (4, 3), (4, 4)}:
                i -= 1
            if i < 0:
                break
        return (_x, _y)

    @functools.lru_cache(maxsize=100)
    def cord_to_number(self, cord):
        (x, y) = cord
        i = 0
        found_it = False
        while not found_it:
            (_x, _y) = self.number_to_cord(i)
            found_it = _x == x and _y == y
            i += 1
            assert i <= N * N - 4
        return i - 1

    def action_to_move(self, action):
        (x, y) = self.number_to_cord(action)
        return self.number_to_move(x, y)