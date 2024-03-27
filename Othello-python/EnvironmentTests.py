from constants import WHITE, BLACK, N, NONE, BLOCKING
import Environments
import othello_wrapper
import numpy as np
import random
import tensorflow as tf
import unittest

env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
}

ACCEPTABLE_ERROR = 1e-5

class EnvironmentTests(unittest.TestCase):
    def test_playthrough(self):
        """Test for playthrough and no crashes"""
        random.seed(1)
        tf.random.set_seed(1)

        e = Environments.OthelloEnvironment(env_parameters)
        n = 100
        cum_r = 0

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
                cum_r += r
            if r > 0.1:
                self.assertGreater(self.get_piece_difference(e, e.colour), 0)
            elif r < -0.1:
                self.assertLess(self.get_piece_difference(e, e.colour), 0)
            else:
                self.assertEqual(self.get_piece_difference(e, e.colour), 0)
            
        
        avg_r = cum_r / n
        self.assertLessEqual(abs(avg_r), 0.2)

    def test_draw(self):
        random.seed(1)
        tf.random.set_seed(1)

        e = Environments.OthelloEnvironment(env_parameters)
        n = 100
        cum_r = 0
        tested1 = False
        tested2 = False

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
                cum_r += r
            if r < 0.1 and r > -0.1:
                self.assertEqual(self.count_pieces(e, WHITE), self.count_pieces(e, BLACK))
                tested1 = True
            if self.count_pieces(e, WHITE) == self.count_pieces(e, BLACK):
                self.assertAlmostEqual(r, 0.0)
                tested2 = True
        self.assertTrue(tested1)
        self.assertTrue(tested2)

    def test_random_moves_made(self):
        """Test for random moves before beginning"""
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 10,
            "epsilon": 0.00
        }

        random.seed(1)
        tf.random.set_seed(1)
        e = Environments.OthelloEnvironment(env_parameters)
        game = othello_wrapper.OthelloGame()
        game.reset()
        default_state = game.board_to_tensor(BLACK)
        n_diff_start_states = 0

        n = 100
        for i in range(n):
            e.reset()
            start_state = e.game.board_to_tensor(e.game.get_player_turn())
            if tf.reduce_sum(tf.abs(start_state - default_state)) > ACCEPTABLE_ERROR or e.game.get_player_turn() == WHITE:
                n_diff_start_states += 1
        self.assertGreaterEqual(n_diff_start_states, 80)

    def test_random_moves_made_reset(self):
        """Test that large amounts of random moves do not crash the game"""

        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 40,
            "epsilon": 0.00
        }
        random.seed(1)
        e = Environments.OthelloEnvironment(env_parameters)
        n = 100
        tf.random.set_seed(1)
        for i in range(n):
            e.reset()

    def test_reset(self):
        random.seed(2)
        tf.random.set_seed(2)

        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }
        e = Environments.OthelloEnvironment(env_parameters)
        e2 = Environments.OthelloEnvironment(env_parameters)
        (s, _, _) = e2.step(tf.ones((1, N, N, 1)))
        self.assertGreaterEqual(tf.reduce_sum(tf.abs(s - e.state)), ACCEPTABLE_ERROR)
        e2.reset()
        self.assertLessEqual(tf.reduce_sum(tf.abs(e2.state - e.state)), ACCEPTABLE_ERROR)

    def test_reset_learning(self):
        random.seed(2)
        tf.random.set_seed(2)

        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }
        e = Environments.OthelloLearningEnvironment(env_parameters)
        e2 = Environments.OthelloLearningEnvironment(env_parameters)
        (s, _, _) = e2.step(tf.ones((1, N, N, 1)))
        self.assertGreaterEqual(tf.reduce_sum(tf.abs(s - e.state)), ACCEPTABLE_ERROR)
        e2.reset()
        self.assertLessEqual(tf.reduce_sum(tf.abs(e2.state - e.state)), ACCEPTABLE_ERROR)

    def test_step_1(self):
        random.seed(1)
        tf.random.set_seed(1)

        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.OthelloEnvironment(env_parameters) #Black
        board = e.game.board_to_tensor(BLACK)
        action = np.zeros((1, N, N, 3))
        action[0, 4, 2, 0] = 1
        action = tf.convert_to_tensor(action)
        (s, r, d) = e.step(action)
        self.assertGreaterEqual(tf.reduce_sum(tf.abs(s - board)), ACCEPTABLE_ERROR)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(d, 0.0)

    def test_halfcheetah(self):
        env_parameters = {
            "render_mode": None,
            "seed": 1
        }
        env = Environments.HalfCheetahEnv(env_parameters)
        action = tf.clip_by_value(tf.random.normal((1, 6)), -1.0, 1.0)
        d = 0.0
        i = 0
        while d < 0.999:
            (s, r, d) = env.step(action)
            self.assertEqual(s.shape, (1, 17))
            self.assertEqual(r.shape, ())
            i += 1
            if d < 0.999:
                self.assertAlmostEqual(d, 0.0)
        self.assertAlmostEqual(d, 1.0)
        self.assertEqual(i, 1000)

    def get_piece_difference(self, env, perspective):
        diff = 0
        for x in range(N):
            for y in range(N):
                square = env.game.game.board[x][y]
                if square == NONE or square == BLOCKING:
                    pass
                elif square == perspective:
                    diff += 1
                else:
                    diff -= 1
        return diff
    
    def count_pieces(self, env, colour):
        num = 0
        for x in range(N):
            for y in range(N):
                square = env.game.game.board[x][y]
                if square == colour:
                    num += 1
        return num

    def test_inverse(self):
        random.seed(1)
        tf.random.set_seed(1)
        
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.OthelloInverseEnvironment(env_parameters)
        n = 100
        cum_r = 0

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
                cum_r += r
            if r > 0.1:
                self.assertLess(self.get_piece_difference(e, e.colour), 0)
            elif r < -0.1:
                self.assertGreater(self.get_piece_difference(e, e.colour), 0)
            else:
                self.assertEqual(self.get_piece_difference(e, e.colour), 0)
        
        avg_r = cum_r / n
        self.assertLessEqual(abs(avg_r), 0.2)

    def test_score(self):
        random.seed(1)
        tf.random.set_seed(1)
        
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.OthelloScoreEnvironment(env_parameters)
        n = 100
        cum_r = 0

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
                cum_r += r
            self.assertAlmostEqual(self.get_piece_difference(e, e.colour), r)
        
        avg_r = cum_r / n
        self.assertLessEqual(abs(avg_r), n)

    def test_starting_position_change(self):
        random.seed(1)
        tf.random.set_seed(1)
        
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.OthelloStartingPositionChange(env_parameters)
        e.reset()
        board = e.game.game.board
        self.assertEqual(board[2][3], BLACK)
        self.assertEqual(board[3][2], BLACK)
        self.assertEqual(board[4][5], BLACK)
        self.assertEqual(board[5][4], BLACK)

        self.assertEqual(board[4][2], WHITE)
        self.assertEqual(board[5][3], WHITE)
        self.assertEqual(board[2][4], WHITE)
        self.assertEqual(board[3][5], WHITE)

        self.assertEqual(self.count_pieces(e, BLACK), 6)
        self.assertEqual(self.count_pieces(e, WHITE), 6)
        self.assertEqual(self.count_pieces(e, BLOCKING), 0)
        self.assertEqual(self.count_pieces(e, NONE), 52)

    def test_4_by_4(self):
        random.seed(1)
        tf.random.set_seed(1)
        
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.Othello_four_by_four(env_parameters)
        n = 100

        self.assertEqual(self.count_pieces(e, BLOCKING), 48)
        self.assertEqual(self.count_pieces(e, NONE), 12)
        self.assertEqual(self.count_pieces(e, WHITE), 2)
        self.assertEqual(self.count_pieces(e, BLACK), 2)

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
            self.assertLessEqual(self.count_pieces(e, BLACK) + self.count_pieces(e, WHITE), 16)
            self.assertLessEqual(self.count_pieces(e, NONE), 11)
            self.assertEqual(self.count_pieces(e, BLOCKING), 48)

    def test_6_by_6(self):
        random.seed(1)
        tf.random.set_seed(1)
        
        env_parameters = {
            "high_network_weights_directory": "HIRO_high_weights",
            "low_network_weights_directory": "HIRO_low_weights",
            "h": 1,
            "previous_opponent_prob": 0.2,
            "max_random_moves_before_game": 0,
            "epsilon": 0.00
        }

        e = Environments.Othello_six_by_six(env_parameters)
        n = 100

        self.assertEqual(self.count_pieces(e, BLOCKING), 28)
        self.assertEqual(self.count_pieces(e, NONE), 32)
        self.assertEqual(self.count_pieces(e, WHITE), 2)
        self.assertEqual(self.count_pieces(e, BLACK), 2)

        for i in range(n):
            e.reset()
            d = 0.0
            while d < 0.999:
                action = tf.random.normal((1, N, N, 1))
                (s, r, d) = e.step(action)
            self.assertLessEqual(self.count_pieces(e, BLACK) + self.count_pieces(e, WHITE), 36)
            self.assertLessEqual(self.count_pieces(e, NONE), 33)
            self.assertEqual(self.count_pieces(e, BLOCKING), 28)

if __name__ == '__main__':
    unittest.main()