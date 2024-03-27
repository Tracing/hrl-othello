from constants import N, WHITE, BLACK, NONE, GOAL_SIZE
from Models import get_high_network, get_low_network, get_dqn_network
from othello_wrapper import OthelloGame
import othello
import numpy as np
import random
import tensorflow as tf

class Agent:
    def __init__(self):
        self.game_number = 0
        self.do_logging = False
        self.log_filename = None
        self.colour = None

    def new_game(self, game: OthelloGame, colour):
        self.game_number += 1
        self.log_filename = "Game-{}".format(self.game_number)
        self.colour = colour

    def set_log_directory(self, log_directory):
        self.log_directory = log_directory
        self.do_logging = True

    def set_logging_off(self, log_directory=None):
        self.log_directory = log_directory
        self.do_logging = False

    def get_move(self, game: OthelloGame):
        return random.choice(game.get_moves_2())

    def get_action(self, game: OthelloGame):
        #Might be wrong, needs testing
        return random.choice(game.get_legal_numbers())

    def record_pre_move(self, game: OthelloGame):
        if self.do_logging:
            with open("{}/{}.log".format(self.log_directory, self.log_filename), "a") as f:
                f.write("-------------------- Game Board ---------------\n")
                f.write("Playing {}\n".format("WHITE" if self.colour == WHITE else "BLACK"))
                f.write(str(game) + "\n")

    def get_name(self):
        return "Random Agent"

class MinimaxAgent(Agent):
    def __init__(self, depth: int):
        super(MinimaxAgent, self).__init__()
        self.depth = depth

    def get_move(self, game: OthelloGame):
        seed = random.randint(1, 4294967295)
        return othello.get_minimax_move(game.game.board, game.get_player_turn(), self.depth, seed)

    def get_name(self):
        return "Minimax Agent d={}".format(self.depth)

class MCTSAgent(Agent):
    def __init__(self, n: int, C: float):
        super(MCTSAgent, self).__init__()
        self.n = n
        self.C = C

    def get_move(self, game: OthelloGame):
        seed = random.randint(1, 4294967295)
        return othello.get_mcts_move(game.game.board, game.get_player_turn(), self.n, self.C, seed)

    def get_name(self):
        return "MCTS Agent n={} C={:.4f}".format(self.n, self.C)

class ArtificialLowAgent(Agent):
    """Decoupled agent that uses the high network to set goals and fulfills them via minimax"""
    def __init__(self, depth, high_weights=None, high_network=None):
        super(ArtificialLowAgent, self).__init__()
        assert high_weights is not None or high_network is not None
        assert not (high_weights is not None and high_network is not None)
        if high_weights is not None:
            self.weights_path = high_weights
            self.high_network = get_high_network()
            self.high_network(tf.zeros((1, N, N, 1)))
            self.high_network.load_weights(self.weights_path)
        else:
            self.weights_path = "Imported"
            self.high_network = high_network

        self.depth = depth

    def new_game(self, game: OthelloGame, colour):
        super(ArtificialLowAgent, self).new_game(game, colour)
        self.c = 0
        self.goal = None

    def record_pre_move(self, game: OthelloGame):
        super(ArtificialLowAgent, self).record_pre_move(game)

    def get_move(self, game: OthelloGame):
        seed = random.randint(1, 4294967295)

        self.goal = transform_to_goal(self.high_network(game.board_to_tensor(game.get_player_turn())), M)
        goal = np.zeros((1, N, N, 1))
        for i in range(N):
            for j in range(N):
                goal[i][j] = self.goal[0][i][j][0]
        move = othello.get_minimax_goal_move(game.game.board, goal, game.get_player_turn(), self.depth, seed)
        return move
        
    def get_name(self):
        return "Decoupled Low Agent {}, depth={}".format(self.weights_path, self.depth)

class DQN_Agent(Agent):
    def __init__(self, epsilon: float, weights=None, network=None):
        super(DQN_Agent, self).__init__()
        if network is None:
            self.weights_path = weights
            self.network = get_dqn_network()

            self.network(tf.zeros((1, N, N, 1)))
            self.network.load_weights(self.weights_path)
        else:
            assert weights is None
            self.weights_path = None
            self.network = network

        self.epsilon = epsilon

    def get_action(self, game: OthelloGame):
        legal_numbers = [i for i in range(N * N - 4) if game.number_is_legal(i)]

        if random.random() < self.epsilon:
            best_action = random.choice(legal_numbers)
        else:
            output = self.network(game.board_to_tensor(game.get_player_turn()))

            legal_actions = game.get_legal_numbers()
            best_action = None
            best_score = float('-inf')
            for i in legal_actions:
                score = float(output[0, i])
                if score > best_score:
                    best_score = score
                    best_action = i
        
        return best_action
        
    def get_move(self, game: OthelloGame):
        best_action = self.get_action(game)
        move = game.action_to_move(best_action)
        return move

    def get_name(self):
        return "DQN Agent"

class HIROAgent(Agent):
    def __init__(self, high_network_weights, low_network_weights, low_epsilon, high_epsilon, ai_low_depth=0, random_seed=1):
        super(HIROAgent, self).__init__()
        self.high_network = get_high_network()
        self.high_network(tf.zeros((1, N, N, 1)))
        self.high_network.load_weights(high_network_weights)

        self.low_network = get_low_network()
        self.low_network([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])
        self.low_network.load_weights(low_network_weights)

        self.low_network_weights = low_network_weights
        self.high_network_weights = high_network_weights
        self.goal = None

        self.ai_low_depth = ai_low_depth
        self.random_seed = random_seed
        self.low_epsilon = low_epsilon
        self.high_epsilon = high_epsilon

    def new_game(self, game: OthelloGame, colour):
        super(HIROAgent, self).new_game(game, colour)

    def update_with_weights(self, high_network_weights, low_network_weights):
        self.high_network.load_weights(high_network_weights)
        self.low_network.load_weights(low_network_weights)                

    def get_goal(self, game):
        output = self.high_network(game.board_to_tensor(game.get_player_turn()))
        if self.high_epsilon > 1e-5:
            noise = tf.clip_by_value(tf.random.normal((1, GOAL_SIZE), stddev=self.high_epsilon), -1000, 1000)
        else:
            noise = 0.0
        high_action = output + noise
        return high_action

    def get_action(self, game: OthelloGame):
        assert self.ai_low_depth == 0
        self.goal = self.get_goal(game)
        legal_numbers = [i for i in range(N * N - 4) if game.number_is_legal(i)]

        if random.random() < self.low_epsilon:
            best_action = random.choice(legal_numbers)
        else:
            output = self.low_network([game.board_to_tensor(game.get_player_turn()), self.goal])

            best_score = float('-inf')
            best_action = random.choice(legal_numbers)

            for number in legal_numbers:
                score = output[0, number]
                if score > best_score:
                    best_score = score
                    best_action = number
        
        return best_action

    def get_action_from_state_goal_pair(self, state, goal, legal_numbers):
        output = self.low_network([state, goal])

        best_score = float('-inf')
        best_action = random.choice(legal_numbers)

        for number in legal_numbers:
            score = output[0, number]
            if score > best_score:
                best_score = score
                best_action = number
        
        return best_action


    def get_move(self, game: OthelloGame):
        if self.ai_low_depth == 0:
            best_action = self.get_action(game)
            move = game.action_to_move(best_action)
        else:
            self.goal = self.get_goal(game)
            goal = self.goal[0, :]
            move = othello.get_minimax_goal_move(game.game.board, goal, game.get_player_turn(), self.ai_low_depth, self.random_seed)
        return move

    def _goal_to_str(self, goal):
        return "coin_value: {:.3f}\nstability_value: {:.3f}\nlanes_owned: {:.3f}\ngood_edges_owned: {:.3f}\ncorners_owned: {:.3f}".format(goal[0, 0], goal[0, 1], goal[0, 2], goal[0, 3], goal[0, 4], goal[0, 5])

    def record_pre_move(self, game: OthelloGame):
        if self.do_logging:
            goal_str_1 = self._goal_to_str(self.goal, game.get_player_turn())
            with open("{}/{}.log".format(self.log_directory, self.log_filename), "a") as f:
                f.write("-----------------------------------------\n")
                f.write("Playing {}\n".format("WHITE" if self.colour == WHITE else "BLACK"))
                f.write("-------------------- Goal ---------------\n")
                f.write(goal_str_1 + "\n")
                f.write("-------------------- Game Board ---------------\n")
                f.write(str(game) + "\n")

    def get_name(self):
        return "HIRO Agent low_network={} high_network={}".format(self.low_network_weights, self.high_network_weights)

class TransferAgent(HIROAgent):
    """Decoupled agent that uses the high network to set goals and the low network to fulfill them."""
    def __init__(self, high_network_weights, low_network_weights, low_epsilon, high_epsilon):
        self.high_network = get_high_network()
        self.high_network(tf.zeros((1, N, N, 1)))
        self.high_network.load_weights(high_network_weights)

        self.low_network = get_low_network()
        self.low_network([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])
        self.low_network.load_weights(low_network_weights)

        self.low_network_weights = low_network_weights
        self.high_network_weights = high_network_weights
        self.goal = None

        self.low_epsilon = low_epsilon
        self.high_epsilon = high_epsilon

        self.goal_weights = None

        self.ai_low_depth = 0
        self.random_seed = 1

    def get_goal(self, game):
        output = self.high_network(game.board_to_tensor(game.get_player_turn()))
        output = tf.clip_by_value(output * self.goal_weights, 0.0, 1.0)

        if self.high_epsilon > 1e-5:
            noise = tf.clip_by_value(tf.random.normal((1, GOAL_SIZE), stddev=self.high_epsilon), -self.c, self.c)
        else:
            noise = 0.0
        high_action = output + noise
        return high_action

    def set_goal_weights(self, goal_weights):
        self.goal_weights = tf.convert_to_tensor(goal_weights, dtype=tf.float32)

    def get_name(self):
        return "Transfer Agent-{}".format(self.high_network_weights)