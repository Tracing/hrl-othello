from Agents import Agent, MinimaxAgent, MCTSAgent, DDPG_Agent, TD3_Agent, HIROAgent
from constants import WHITE, BLACK, N, NONE
import Environments
import math
import random
import tensorflow as tf
import Tournament
import unittest

class TournamentTests(unittest.TestCase):
    def test_parity(self):
        random.seed(1)
        tf.random.set_seed(1)
        agent1s = [Agent(), MinimaxAgent(1), MinimaxAgent(3), MCTSAgent(50, math.sqrt(2))]
        agent2s = [Agent(), MinimaxAgent(1), MinimaxAgent(3), MCTSAgent(50, math.sqrt(2))]
        for (agent1, agent2) in zip(agent1s, agent2s):
            win_ratio = Tournament.play_n_games(200, agent1, agent2, "tmp/1", "tmp/2") / 200
            self.assertGreaterEqual(win_ratio, 0.4)
            self.assertLessEqual(win_ratio, 0.6)

    def test_imparity(self):
        random.seed(1)
        tf.random.set_seed(1)
        agent1s = [Agent(), MinimaxAgent(1)]
        agent2s = [MinimaxAgent(1), MinimaxAgent(3)]
        for (agent1, agent2) in zip(agent1s, agent2s):
            win_ratio = Tournament.play_n_games(200, agent1, agent2, "tmp/1", "tmp/2") / 200
            self.assertGreaterEqual(win_ratio, 0.0)
            self.assertLessEqual(win_ratio, 0.4)

if __name__ == '__main__':
    unittest.main()