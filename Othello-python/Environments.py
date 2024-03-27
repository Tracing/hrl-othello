from Agents import HIROAgent
from constants import N, NULL, WHITE, BLACK, NONE, BLOCKING
from othello_wrapper import OthelloGame
import gymnasium as gym
import math
import os
import othello
import random
import tensorflow as tf

class OthelloEnvironment:
    def auxillary_reward(self):
        tensor = self.game.board_to_tensor(self.colour)
        own_score = tf.reduce_sum(tf.maximum(tensor, 0))
        other_score = tf.reduce_sum(tf.abs(tf.minimum(tensor, 0)))
        return (own_score / (own_score + other_score)) * 0.01

    def __init__(self, parameters):
        self.colour = random.choice([chr(1), chr(2)])
        self.game = OthelloGame()
        self.state = self.game.board_to_tensor(self.colour)
        self.parameters = parameters
        self.reset()

    def reset(self):
        finished = False
        n_moves = random.randint(0, min(self.parameters["max_random_moves_before_game"], N * N - 1))
        while not finished:
            self.colour = random.choice([chr(1), chr(2)])
            self.reset_game()
            self.state = self.game.board_to_tensor(self.colour)
            self.do_random_moves(n_moves)

            while self.colour != self.game.get_player_turn() and not self.game.game_has_ended():
                self.do_other_move()
            
            finished = not self.game.game_has_ended()

    def reset_game(self):
        self.game.reset()

    def update(self):
        pass

    def step(self, action):
        assert self.colour == self.game.get_player_turn()
        assert action in self.get_legal_numbers()
        move = self.game.action_to_move(action)

        self.game.make_move_2(move)

        while (self.colour != self.game.get_player_turn()) and not self.game.game_has_ended():
            self.do_other_move()

        if self.game.game_has_ended():
            reward = 1.0 if self.game.get_winner() == self.colour else 0.0 if self.game.get_winner() in [NONE, BLOCKING] else 0.0
            terminated = 1.0
        else:
            reward = 0.0
            terminated = 0.0

        assert self.colour == self.game.get_player_turn() or self.game.game_has_ended()
        reward = tf.cast(tf.reshape(reward, (1, 1)), dtype='float32')
        terminated = tf.cast(tf.reshape(terminated, (1, 1)), dtype='float32')
        return (self.game.board_to_tensor(self.colour), reward, terminated)

    def do_other_move(self):
        self.game.make_move_2(random.choice(self.game.get_moves_2()))

    def do_random_move(self):
        moves = self.game.get_moves_2()
        move = random.choice(moves)
        self.game.make_move_2(move)

    def do_random_moves(self, n):
        i = 0
        while i < n and not self.game.game_has_ended():
            self.do_random_move()
            i += 1

    def get_legal_numbers(self):
        return self.game.get_legal_numbers()

class OthelloEasyEnvironment(OthelloEnvironment):
    def do_other_move(self):
        best_score = float('-inf')
        best_move = None
        for move in self.game.get_moves_2():
            score = move.y * N + move.x
            if score > best_score:
                best_score = score
                best_move = move
        self.game.make_move_2(best_move)

class OthelloLearningEnvironment(OthelloEnvironment):
    def __init__(self, parameters, opponent_name):
        self.opponent_name = opponent_name
        super(OthelloLearningEnvironment, self).__init__(parameters)

    def reset(self):
        super(OthelloLearningEnvironment, self).reset()

    def step(self, action):
        assert self.colour == self.game.get_player_turn()
        assert action in self.get_legal_numbers()
        
        move = self.game.action_to_move(action)

        self.game.make_move_2(move)

        while (self.colour != self.game.get_player_turn()) and not self.game.game_has_ended():
            self.do_other_move()

        if self.game.game_has_ended():
            reward = 1.0 if self.game.get_winner() == self.colour else 0.5 if self.game.get_winner() in [NONE, BLOCKING] else 0
            terminated = 1.0
        else:
            reward = 0.0
            terminated = 0.0

        assert self.colour == self.game.get_player_turn() or self.game.game_has_ended()
        reward = tf.cast(tf.reshape(reward, (1, 1)), dtype='float32')
        terminated = tf.cast(tf.reshape(terminated, (1, 1)), dtype='float32')
        return (self.game.board_to_tensor(self.colour), reward, terminated)

    def do_other_move(self):
        seed = random.randint(1, 4294967295)
        if self.opponent_name == "random":
            self.game.make_move_2(random.choice(self.game.get_moves_2()))
        elif self.opponent_name == "minimax_1":
            self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 1, seed))
        elif self.opponent_name == "minimax_2":
            self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 2, seed))
        elif self.opponent_name == "minimax_3":
            self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 3, seed))
        else:
            assert self.opponent_name == "mcts_50"
            self.game.make_move_2(othello.get_mcts_move(self.game.game.board, self.game.get_player_turn(), 50, math.sqrt(2), seed))
    
class OthelloScoreEnvironment(OthelloEnvironment):
    def get_score(self):
        a = tf.reduce_sum(tf.clip_by_value(self.game.board_to_tensor(self.colour, True), 0.0, 1.0))
        b = tf.reduce_sum(tf.clip_by_value(-self.game.board_to_tensor(self.colour, True), 0.0, 1.0))
        return a / (a + b)

    def step(self, action):
        assert self.colour == self.game.get_player_turn()
        move = self.game.action_to_move(action)

        self.game.make_move_2(move)

        while (self.colour != self.game.get_player_turn()) and not self.game.game_has_ended():
            self.do_other_move()

        if self.game.game_has_ended():
            reward = self.get_score()
            terminated = 1.0
        else:
            reward = 0.0
            terminated = 0.0

        assert self.colour == self.game.get_player_turn() or self.game.game_has_ended()
        reward = tf.cast(tf.reshape(reward, (1, 1)), dtype='float32')
        terminated = tf.cast(tf.reshape(terminated, (1, 1)), dtype='float32')
        return (self.game.board_to_tensor(self.colour), reward, terminated)

    def do_other_move(self):
        seed = random.randint(1, 4294967295)
        self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 1, seed))
    
class OthelloStartingPositionChange(OthelloEnvironment):
    def reset_game(self):
        self.game.reset()
        self.game.set_square(2, 3, BLACK)
        self.game.set_square(3, 2, BLACK)
        self.game.set_square(4, 5, BLACK)
        self.game.set_square(5, 4, BLACK)

        self.game.set_square(4, 2, WHITE)
        self.game.set_square(5, 3, WHITE)
        self.game.set_square(2, 4, WHITE)
        self.game.set_square(3, 5, WHITE)

    def do_other_move(self):
        seed = random.randint(1, 4294967295)
        self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 1, seed))

class Othello_four_by_four(OthelloEnvironment):
    def reset_game(self):
        self.game.reset()
        for x in range(N):
            for y in range(N):
                if x <= 1:
                    self.game.set_square(x, y, BLOCKING)
                elif x >= 6:
                    self.game.set_square(x, y, BLOCKING)
                elif y <= 1:
                    self.game.set_square(x, y, BLOCKING)
                elif y >= 6:
                    self.game.set_square(x, y, BLOCKING)

    def do_other_move(self):
        seed = random.randint(1, 4294967295)
        self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 1, seed))

class Othello_six_by_six(OthelloEnvironment):
    def reset_game(self):
        self.game.reset()
        for x in range(N):
            for y in range(N):
                if x <= 0:
                    self.game.set_square(x, y, BLOCKING)
                elif x >= 7:
                    self.game.set_square(x, y, BLOCKING)
                elif y <= 0:
                    self.game.set_square(x, y, BLOCKING)
                elif y >= 7:
                    self.game.set_square(x, y, BLOCKING)

    def do_other_move(self):
        seed = random.randint(1, 4294967295)
        self.game.make_move_2(othello.get_minimax_move(self.game.game.board, self.game.get_player_turn(), 1, seed))

class OthelloScoreEnvironmentValidation(OthelloEnvironment):
    def do_other_move(self):
       self.game.make_move_2(random.choice(self.game.get_moves_2()))

class OthelloStartingPositionChangeValidation(OthelloEnvironment):
    def do_other_move(self):
        self.game.make_move_2(random.choice(self.game.get_moves_2()))

class Othello_four_by_fourValidation(OthelloEnvironment):
    def do_other_move(self):
        self.game.make_move_2(random.choice(self.game.get_moves_2()))

class Othello_six_by_sixValidation(OthelloEnvironment):
    def do_other_move(self):
        self.game.make_move_2(random.choice(self.game.get_moves_2()))

class HalfCheetahEnv(OthelloEnvironment):
    def __init__(self, parameters):
        self.parameters = parameters
        self.reset()

    def reset(self):
        self.env = gym.make("HalfCheetah-v4", render_mode=self.parameters["render_mode"])
        (self.state, info) = self.env.reset(seed=self.parameters["seed"])

    def step(self, action):
        assert action in self.get_legal_numbers()

        (self.state, reward, env_terminated, env_truncated, info) = self.env.step(tf.reshape(action, (6,)))
        reward = tf.cast(tf.reshape(reward, (1, 1)), dtype='float32')

        terminated = 1.0 if env_terminated or env_truncated else 0.00
        terminal = env_terminated or env_truncated

        if terminal:
            self.env.close()

        return (self.board_to_tensor(), reward, terminated)
    
    def board_to_tensor(self):
        return tf.cast(tf.reshape(self.state, (1, 17)), dtype='float32')
    
class PendulumEnv(OthelloEnvironment):
    def __init__(self, parameters):
        self.parameters = parameters
        self.reset()

    def reset(self):
        self.env = gym.make("Pendulum-v1", render_mode=self.parameters["render_mode"])
        (self.state, info) = self.env.reset(seed=self.parameters["seed"])

    def step(self, action):
        assert action in self.get_legal_numbers()
        
        (self.state, reward, env_terminated, env_truncated, info) = self.env.step(tf.reshape(action, (-1,)))
        reward = tf.cast(tf.reshape(reward, (1, 1)), dtype='float32')

        terminated = 1.0 if env_terminated or env_truncated else 0.00
        terminal = env_terminated or env_truncated

        if terminal:
            self.env.close()

        terminated = tf.cast(tf.reshape(terminated, (1, 1)), dtype='float32')

        return (self.board_to_tensor(), reward, terminated)
    
    def board_to_tensor(self):
        return tf.cast(tf.reshape(self.state, (1, -1)), dtype='float32')
    