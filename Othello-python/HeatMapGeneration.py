from constants import WHITE, BLACK, N, NONE, GOAL_SIZE
from Models import get_high_network, get_high_network_critic, get_low_network
import numpy as np
import othello_wrapper
import random
import tensorflow as tf
import keract

def generate_data(N, goal=tf.ones((1, GOAL_SIZE))):
    game = othello_wrapper.OthelloGame()
    xs = []
    for i in range(N):
        game.reset()
        j = 0
        while not game.game_has_ended():
            game.make_move_2(random.choice(game.get_moves_2()))
            if j > 0:
                xs.append(game.board_to_tensor(colour=random.choice([WHITE, BLACK])))
            j += 1
    i = random.randint(0, len(xs)-1)
    print("Taking board {}/{}".format(i+1, len(xs)))
    xs = xs[i]
    return (xs, [xs, goal])

def initialize_networks(weights_paths):
    #Initialize actor and critic networks and target networks and initialize parameters
    low_network = get_low_network()
    low_network_target = get_low_network()

    high_network = get_high_network()
    high_network_target = get_high_network()
    high_network_critic = get_high_network_critic()
    high_network_critic_target = get_high_network_critic()

    dummy_input = tf.zeros((1, N, N, 1))
    dummy_input2 = tf.zeros((1, GOAL_SIZE))

    low_network([dummy_input, dummy_input2])
    low_network_target([dummy_input, dummy_input2])
    high_network(dummy_input)
    high_network_target(dummy_input)

    high_network_critic([dummy_input, dummy_input2])
    high_network_critic_target([dummy_input, dummy_input2])


    #low_network.load_weights(weights_paths["low_weights_path"])
    #low_network_target.load_weights(weights_paths["low_weights_targets_path"])
    high_network.load_weights(weights_paths["high_weights_path"])
    #high_network_target.load_weights(weights_paths["high_weights_targets_path"])

    networks = {"high_network": high_network,
                "low_network": low_network,
                "high_network_critic": high_network_critic,

                "high_network_target": high_network_target,
                "low_network_target": low_network_target,
                "high_network_critic_target": high_network_critic_target
                }

    return networks

def generate_heatmap(data_N, directory, weights_paths, network_name, goal=tf.ones((1, GOAL_SIZE)), seed=None):
    if not seed is None:
        random.seed(seed)

    (data1, data2) = generate_data(data_N, goal)
    
    networks = initialize_networks(weights_paths)
    assert network_name in networks.keys()

    for (name, network) in networks.items():
        if name == network_name:
            if name in {"high_network", "high_network_target"}:
                data = data1
            elif name in {"low_network", "low_network_target"}:
                data = data2
            elif name in {"high_network_critic", "high_network_critic_target"}:
                data = data2
            else:
                assert False

            print("Doing {}".format(name))

            output = keract.get_activations(network, data, auto_compile=True)
            keract.display_activations(output, save=True, directory=directory)

def display_output(directory, output):
    pass

if __name__ == "__main__":
    output_dir = "keract_output"
    weights_dir = "HIRO_weights"
    network_num = 9
    network = "high_network"
    goal = np.ones((1, GOAL_SIZE))
    goal = tf.convert_to_tensor(goal)

    weights_paths = {
        "high_weights_path": "{}/high_network-{}.h5".format(weights_dir, network_num),
        "low_weights_path": "{}/low_network-{}.h5".format(weights_dir, network_num),
        "low_weights_targets_path": "{}/low_network_target-{}.h5".format(weights_dir, network_num),
        "high_weights_targets_path": "{}/high_network_target-{}.h5".format(weights_dir, network_num),
        "high_critic_weights_path": "{}/high_network_critic-{}.h5".format(weights_dir, network_num),
        "high_critic_weights_target_path": "{}/high_network_critic_target-{}.h5".format(weights_dir, network_num)
    }

    generate_heatmap(1, output_dir, weights_paths, network, goal=goal, seed=2)