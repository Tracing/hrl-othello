from constants import WHITE, BLACK, N, NONE
import numpy as np
import othello_wrapper
import tensorflow as tf
import unittest

ACCEPTABLE_ERROR = 1e-5

class OthelloWrapperTests(unittest.TestCase):
    def test_board_to_tensor_1(self):
        game = othello_wrapper.OthelloGame()
        white_tensor = game.board_to_tensor(WHITE)
        black_tensor = game.board_to_tensor(BLACK)

        target_black_tensor = np.zeros((1, N, N, 3), dtype=np.int64)
        target_black_tensor[0, 3, 3, 0] = 1
        target_black_tensor[0, 4, 4, 0] = 1

        target_black_tensor[0, 4, 3, 1] = 1
        target_black_tensor[0, 3, 4, 1] = 1

        target_black_tensor[0, 2, 4, 2] = 1
        target_black_tensor[0, 3, 5, 2] = 1
        target_black_tensor[0, 4, 2, 2] = 1
        target_black_tensor[0, 5, 3, 2] = 1

        target_black_tensor = tf.convert_to_tensor(target_black_tensor)

        self.assertLessEqual(tf.reduce_sum(target_black_tensor - black_tensor), ACCEPTABLE_ERROR)

        target_white_tensor = np.zeros((1, N, N, 3), dtype=np.int64)
        target_white_tensor[0, 4, 3, 0] = 1
        target_white_tensor[0, 3, 4, 0] = 1

        target_white_tensor[0, 3, 3, 1] = 1
        target_white_tensor[0, 4, 4, 1] = 1

        target_white_tensor[0, 2, 4, 2] = 1
        target_white_tensor[0, 3, 5, 2] = 1
        target_white_tensor[0, 4, 2, 2] = 1
        target_white_tensor[0, 5, 3, 2] = 1

        target_white_tensor = tf.convert_to_tensor(target_white_tensor)

        self.assertLessEqual(tf.reduce_sum(tf.abs(target_white_tensor - white_tensor)), ACCEPTABLE_ERROR)

    def test_reset(self):
        game = othello_wrapper.OthelloGame()
        new_game = othello_wrapper.OthelloGame()
        game.make_move_2(game.number_to_move(4, 2))
        self.assertNotEqual(game.get_player_turn(), new_game.get_player_turn())
        self.assertGreater(tf.reduce_sum(tf.abs(game.board_to_tensor(WHITE) - new_game.board_to_tensor(WHITE))), ACCEPTABLE_ERROR)

    def test_get_moves(self):
        game = othello_wrapper.OthelloGame()
        (whiteMoves, blackMoves) = game.get_moves()
        self.assertEqual(whiteMoves[0].x, -1)
        self.assertEqual(whiteMoves[0].y, -1)
        self.assertEqual(len(whiteMoves), 1)

        cords = [(move.x, move.y) for move in blackMoves]
        for i in range(len(blackMoves)):
            self.assertIn((blackMoves[i].x, blackMoves[i].y), cords)
        self.assertEqual(len(blackMoves), 4)

    def test_get_moves_2(self):
        game = othello_wrapper.OthelloGame()
        moves = game.get_moves_2()

        cords = [(move.x, move.y) for move in moves]
        for i in range(len(moves)):
            self.assertIn((moves[i].x, moves[i].y), cords)
        self.assertEqual(len(moves), 4)

    def test_make_move_1(self):
        game = othello_wrapper.OthelloGame()
        game.make_move(game.number_to_move(-1, -1), game.number_to_move(2, 4))
        self.assertEqual(game.get_player_turn(), WHITE)
        self.assertEqual(game.game.at_square(2, 4), BLACK)
        self.assertEqual(game.game.at_square(3, 4), BLACK)
        self.assertEqual(game.game.at_square(4, 4), BLACK)        

    def test_make_move_2(self):
        game = othello_wrapper.OthelloGame()
        game.make_move_2(game.number_to_move(2, 4))
        self.assertEqual(game.get_player_turn(), WHITE)
        self.assertEqual(game.game.at_square(2, 4), BLACK)
        self.assertEqual(game.game.at_square(3, 4), BLACK)
        self.assertEqual(game.game.at_square(4, 4), BLACK)
        
    def test_get_legal_numbers_1(self):
        game = othello_wrapper.OthelloGame()
        legal_numbers = game.get_legal_numbers()
        self.assertSequenceEqual(legal_numbers, [(2, 4), (3, 5), (4, 2), (5, 3)])

    def test_number_to_move(self):
        game = othello_wrapper.OthelloGame()
        move = game.number_to_move(7, 7)
        self.assertEqual(move.x, 7)
        self.assertEqual(move.y, 7)

    def test_get_player_turn(self):
        game = othello_wrapper.OthelloGame()
        self.assertEqual(game.get_player_turn(), BLACK)
        game.make_move_2(game.get_moves_2()[0])
        self.assertEqual(game.get_player_turn(), WHITE)

    def test_number_is_legal(self):
        game = othello_wrapper.OthelloGame()
        self.assertTrue(game.number_is_legal(2, 4))
        self.assertFalse(game.number_is_legal(5, 5))

    def test_get_winner_and_game_has_ended(self):
        game = othello_wrapper.OthelloGame()
        self.assertEqual(game.get_winner(), NONE)
        self.assertFalse(game.game_has_ended())
        
        while not game.game_has_ended():
            move = game.get_moves_2()[0]
            game.make_move_2(move)

        self.assertIn(game.get_winner(), [WHITE, BLACK])
        self.assertTrue(game.game_has_ended())

    def test_action_to_number_and_move(self):
        game = othello_wrapper.OthelloGame()
        action = np.zeros((1, N, N, 1))
        action[0, 0, 0, 0] = 100
        action[0, 3, 5, 0] = 10
        action = tf.convert_to_tensor(action)
        number = game.action_to_number(action)
        self.assertEqual(number[0], 3)
        self.assertEqual(number[1], 5)
        move = game.action_to_move(action)
        self.assertEqual(move.x, 3)
        self.assertEqual(move.y, 5)

    def test_set_square(self):
        game = othello_wrapper.OthelloGame()
        self.assertEqual(game.game.at_square(0, 0), NONE)
        game.set_square(0, 0, BLACK)
        self.assertEqual(game.game.at_square(0, 0), BLACK)

        self.assertEqual(game.game.at_square(5, 5), NONE)
        game.set_square(5, 5, WHITE)
        self.assertEqual(game.game.at_square(5, 5), WHITE)

        self.assertNotEqual(game.game.at_square(3, 3), NONE)
        game.set_square(3, 3, NONE)
        self.assertEqual(game.game.at_square(3, 3), NONE)

if __name__ == '__main__':
    unittest.main()