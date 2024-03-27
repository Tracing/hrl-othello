from Agents import HIROAgent, MCTSAgent, Agent
from constants import WHITE, BLACK, N, GOAL_SIZE
from keras.utils.np_utils import to_categorical
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, differential_entropy, zscore
import math
import numpy as np
import tensorflow as tf
import othello_wrapper
import random

def generate_dataset(n_games, n_items, high_network_weights, low_network_weights, agent):
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)
    game = othello_wrapper.OthelloGame()

    states = []
    actions = []
    goals = []
    legal_numbers = []

    for i in range(n_games):
        perspective = random.choice([WHITE, BLACK])
        game.reset()

        while not game.game_has_ended():
            states.append(game.board_to_tensor(perspective))
            goal = agent.get_goal(game)
            goals.append(goal)
            legal_numbers.append(game.get_legal_numbers())
            action = agent.get_action_from_state_goal_pair(states[-1], goals[-1], legal_numbers[-1])
            actions.append(action)

            game.make_move_2(random.choice(game.get_moves_2()))
        
        print("Generated data for game {}/{}".format(i+1, n_games))

    states = tf.concat(states[:n_items], 0)
    actions = tf.convert_to_tensor(actions[:n_items])
    goals = tf.concat(goals[:n_items], 0)
    legal_numbers = legal_numbers[:n_items]

    return (states, actions, goals, legal_numbers)

def generate_dataset_2(n_games, n_items, high_network_weights, low_network_weights):
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)
    agent2 = MCTSAgent(50, math.sqrt(2))
    game = othello_wrapper.OthelloGame()

    states = []
    actions = []
    goals = []
    legal_numbers = []

    for i in range(n_games):
        perspective = random.choice([WHITE, BLACK])
        game.reset()

        while not game.game_has_ended():
            states.append(game.board_to_tensor(perspective))
            goal = agent.get_goal(game)
            goals.append(goal)
            legal_numbers.append(game.get_legal_numbers())
            action = agent.get_action_from_state_goal_pair(states[-1], goals[-1], legal_numbers[-1])
            actions.append(action)

            if game.get_player_turn() == perspective:
                game.make_move_2(agent.get_move(game))
            else:
                game.make_move_2(agent2.get_move(game))
        
        print("Generated data for game {}/{}".format(i+1, n_games))

    states = tf.concat(states[:n_items], 0)
    actions = tf.convert_to_tensor(actions[:n_items])
    goals = tf.concat(goals[:n_items], 0)
    legal_numbers = legal_numbers[:n_items]

    return (states, actions, goals, legal_numbers)

def generate_dataset_3():
    agent = Agent()
    game = othello_wrapper.OthelloGame()

    states = []
    legal_numbers = []

    game.reset()
    game.make_move_2(agent.get_move(game))

    states.append(game.board_to_tensor(WHITE))
    legal_numbers.append(game.get_legal_numbers())
        
    states = tf.concat(states, 0)

    return (states, legal_numbers)

def run_experiments(seed, n_games, n_items, N, high_network_weights, low_network_weights):
    random.seed(seed)
    tf.random.set_seed(seed) 

    (states, actions, goals, legal_numbers) = generate_dataset(n_games, n_items, high_network_weights, low_network_weights)
    record_goals(goals)
    run_experiment_2(N, high_network_weights, low_network_weights, states, actions, goals, legal_numbers)

    print("All done!")

def record_goals(goals, outfile_name="./information_gain_goals.csv"):
    goals_str = []
    for i in range(len(goals)):
        l = []
        for j in range(GOAL_SIZE):
            l.append("{:.3f}".format(goals[i, j]))
        goals_str.append(",".join(l))
        goals_str.append("\n")
    
    with open(outfile_name, "w") as f:
        f.writelines(goals_str)
 
def run_experiment_2(N, high_network_weights, low_network_weights, states, actions, goals, legal_numbers):
    assert len(states) % N == 0
    print(len(states))
    step = len(states) // N
    proportions_of_actions_changed = []
    for i in range(N):
        print("Running epoch {}/{}".format(i+1, N))
        j = step * i
        k = step * (i + 1)
        proportions_of_actions_changed.append(one_epoch_experiment_2(high_network_weights, low_network_weights, states[j:k], actions[j:k], goals[j:k], legal_numbers[j:k]))
    
    result = ttest_1samp(proportions_of_actions_changed, 0.0, alternative='greater')
    p_value = result.pvalue
    confidence_interval = result.confidence_interval()

    print(proportions_of_actions_changed)
    print(p_value)
    print(np.mean(proportions_of_actions_changed))
    print(confidence_interval)

def one_epoch_experiment_2(high_network_weights, low_network_weights, states, actions, goals, legal_numbers):
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)
    indices = tf.range(len(goals))
    indices = tf.random.shuffle(indices)
    new_goals = tf.gather(goals, indices)
    new_actions = []
    for i in range(len(states)):
        new_action = agent.get_action_from_state_goal_pair(states[i:i+1], new_goals[i:i+1], legal_numbers[i])
        new_actions.append(new_action)

    new_actions = tf.convert_to_tensor(new_actions)

    proportion_of_actions_changed = tf.reduce_sum(tf.math.minimum(tf.abs(actions - new_actions), 1) / len(new_actions))
    return float(proportion_of_actions_changed)

def one_addition(high_network_weights, low_network_weights, states, actions, goals, legal_numbers):
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)
    new_goals = tf.clip_by_value(goals + tf.random.uniform(goals.shape, -1, 1), 0.1, 1)

    record_goals(new_goals, "./information_gain_new_goals.csv")

    new_actions = []
    for i in range(len(states)):
        new_action = agent.get_action_from_state_goal_pair(states[i:i+1], new_goals[i:i+1], legal_numbers[i])
        new_actions.append(new_action)

    new_actions = tf.convert_to_tensor(new_actions)

    proportion_of_actions_changed = tf.reduce_sum(tf.math.minimum(tf.abs(actions - new_actions), 1) / len(new_actions))
    return float(proportion_of_actions_changed)

def static_goals_addition(high_network_weights, low_network_weights, states, actions, legal_numbers):
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)
    new_goals = []
    for i in range(32):
        new_goals.append(tf.reshape(tf.convert_to_tensor([max(float(j), 0.1) for j in list(str(bin(i))[2:].rjust(5, "0"))]), (1, 5)))

    new_actions = []
    for i in range(len(states)):
        new_action = agent.get_action_from_state_goal_pair(states[i:i+1], random.choice(new_goals), legal_numbers[i])
        new_actions.append(new_action)

    new_actions = tf.convert_to_tensor(new_actions)

    proportion_of_actions_changed = tf.reduce_sum(tf.math.minimum(tf.abs(actions - new_actions), 1) / len(new_actions))
    return float(proportion_of_actions_changed)

def run_additions(seed, n_games, n_items, N, high_network_weights, low_network_weights):
    (states, actions, goals, legal_numbers) = generate_dataset_2(n_games, n_items, high_network_weights, low_network_weights)
    record_goals(goals)

    proportions_str = [str(one_addition(high_network_weights, low_network_weights, states, actions, goals, legal_numbers)) for _ in range(N)]

    s = ", ".join(proportions_str)
    
    with open("./information_gain_new_proportions.csv", "w") as f:
        f.write(s)

def run_static_goals_addition(seed, n_games, n_items, N, high_network_weights, low_network_weights):
    (states, actions, goals, legal_numbers) = generate_dataset_2(n_games, n_items, high_network_weights, low_network_weights)
    record_goals(goals)

    proportions_str = [str(static_goals_addition(high_network_weights, low_network_weights, states, actions, legal_numbers)) for _ in range(N)]

    s = ", ".join(proportions_str)
    
    with open("./information_gain_new_proportions_static_goals.csv", "w") as f:
        f.write(s)

def run_single_position_addition(high_network_weights, low_network_weights):
    (states, legal_numbers) = generate_dataset_3()
    agent = HIROAgent(high_network_weights, low_network_weights, 0.0, 0.0)

    new_goals = []
    for i in range(32):
        new_goals.append(tf.reshape(tf.convert_to_tensor([max(float(j), 0.1) for j in list(str(bin(i))[2:].rjust(5, "0"))]), (1, 5)))

    actions = []
    for goal in new_goals:
        action = agent.get_action_from_state_goal_pair(states, goal, legal_numbers[0])
        actions.append(str(action))

    s = ", ".join(actions)

    with open("./information_gain_single_position_actions.csv", "w") as f:
        f.write(s)

def run_goals_for_many(n_seeds, n_games, n_items):
    for seed in range(1, n_seeds+1):
        high_network_weights = "HIRO_weights_30_seeds/HIRO_{}/high_network-1.h5".format(seed)
        low_network_weights = "HIRO_weights_30_seeds/HIRO_{}/low_network-1.h5".format(seed)

        random.seed(seed)
        tf.random.set_seed(seed) 

        (states, actions, goals, legal_numbers) = generate_dataset_2(n_games, n_items, high_network_weights, low_network_weights)
        record_goals(goals, outfile_name="goals_{}.csv".format(seed))

if __name__ == "__main__":
    #run_test(1, 5000, "drive/MyDrive/othello/information_gain_weights/high_network.h5", "drive/MyDrive/othello/information_gain_weights/low_network.h5")
    #run_additions(1, 500, 10000, 50, "information_gain_weights/high_network-9.h5", "information_gain_weights/low_network-9.h5")
    #run_static_goals_addition(1, 500, 10000, 50, "information_gain_weights/high_network-9.h5", "information_gain_weights/low_network-9.h5")
    #run_single_position_addition("information_gain_weights/high_network-9.h5", "information_gain_weights/low_network-9.h5")
    #run_experiments(1, 500, 10000, 1000, "information_gain_weights/high_network-9.h5", "information_gain_weights/low_network-9.h5")
    run_goals_for_many(30, 10, 75)
