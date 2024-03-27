from Environments import Othello_four_by_four
import random
import tensorflow as tf

env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "max_random_moves_before_game": 0
}


e = Othello_four_by_four(env_parameters)
e.reset()

finished = False
while not finished:
    print(e.game)
    (_, r, d) = e.step(random.choice(e.game.get_legal_numbers()))
    finished = d > 0.5
print(r)