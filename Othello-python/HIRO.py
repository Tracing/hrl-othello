from Agents import HIROAgent, Agent, MinimaxAgent, MCTSAgent
from constants import N, NULL, WHITE, BLACK, NONE, DRAW, GOAL_SIZE
from misc_functions import wipe_dir, state_to_high_action, choose_env_i
from ResultsTableProduction import create_results_HIRO, write_results
import collections
import gc
import os
import othello
from othello_wrapper import OthelloGame
from ReplayBuffer import ReplayBuffer
import math
import misc_functions
import numpy as np
import tensorflow as tf
import time
import Tournament
import random

def record_epoch_history(summary_writer, last_env_i, env_rs, steps):
    with summary_writer.as_default():
        tf.summary.scalar('training_level', last_env_i, step=steps[7])
        tf.summary.scalar('last_env_mean_reward', np.mean(env_rs[last_env_i]), step=steps[7])

def update_moving_averages(env_rs, last_env_i, cum_r, meta_parameters):
    env_rs[last_env_i].append(cum_r)
    if len(env_rs[last_env_i]) > meta_parameters["student_required_n"]:
        env_rs[last_env_i].popleft()

def save_networks(networks, network_N, meta_parameters):
    networks["low_network"].save_weights("{}/low_network-{}.h5".format(meta_parameters["weights_dir"], network_N))
    networks["low_network_target"].save_weights("{}/low_network_target-{}.h5".format(meta_parameters["weights_dir"], network_N))
    networks["high_network"].save_weights("{}/high_network-{}.h5".format(meta_parameters["weights_dir"], network_N))
    networks["high_network_critic"].save_weights("{}/high_network_critic-{}.h5".format(meta_parameters["weights_dir"], network_N))
    networks["high_network_target"].save_weights("{}/high_network_target-{}.h5".format(meta_parameters["weights_dir"], network_N))
    networks["high_network_critic_target"].save_weights("{}/high_network_critic_target-{}.h5".format(meta_parameters["weights_dir"], network_N))
    network_N += 1

    return network_N

def load_low_weights(networks, low_parameters):
    networks["low_network"].load_weights(low_parameters["start_weights_filepaths"][0])
    networks["low_network"]([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])
    networks["low_network_target"].load_weights(low_parameters["start_weights_filepaths"][1])
    networks["low_network_target"]([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])

def load_high_weights(networks, high_parameters):
    networks["high_network"].load_weights(high_parameters["start_weights_filepaths"][0])
    networks["high_network"](tf.zeros((1, N, N, 1)))
    networks["high_network_target"].load_weights(high_parameters["start_weights_filepaths"][1])
    networks["high_network_target"](tf.zeros((1, N, N, 1)))
    networks["high_network_critic"].load_weights(high_parameters["start_weights_filepaths"][2])
    networks["high_network_critic"]([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])
    networks["high_network_critic_target"].load_weights(high_parameters["start_weights_filepaths"][3])
    networks["high_network_critic_target"]([tf.zeros((1, N, N, 1)), tf.zeros((1, GOAL_SIZE))])

def do_garbage_collection():
    gc.collect()
    tf.keras.backend.clear_session()

def do_evaluation(meta_parameters, networks, high_parameters, low_parameters, evaluation_agent):
    networks["high_network"].save_weights("./high_network_weights.h5")
    networks["low_network"].save_weights("./low_network_weights.h5")
    high_weights_path = "./high_network_weights.h5"
    low_weights_path = "./low_network_weights.h5"
    hiro_agent = HIROAgent(high_weights_path, low_weights_path, low_parameters["epsilon"], high_parameters["epsilon"], meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
    agent = evaluation_agent
    (score, stddev) = Tournament.play_n_games(meta_parameters["opponent_evaluation_games"], hiro_agent, agent, "./tmp/1", "./tmp/2", include_stddev=True)

    time.sleep(0.1)

    with open("{}/data.csv".format(meta_parameters["evaluation_output_dir"]), "a") as f:
        f.write("{:.2f}, {}\n".format(score, stddev))

    time.sleep(0.1)

def do_transfer_evaluation(env, networks, meta_parameters, low_parameters, high_parameters):
    rewards = []
    for _ in range(meta_parameters["opponent_evaluation_games"]):
        rewards.append(float(do_one_transfer_evaluation(env, networks, meta_parameters, low_parameters, high_parameters)))
    rewards_str = ",".join([str(r) for r in rewards])

    time.sleep(0.1)

    with open("{}/data.csv".format(meta_parameters["evaluation_output_dir"]), "a") as f:
        f.write("{}\n".format(rewards_str))
    
    time.sleep(0.1)

def do_one_transfer_evaluation(env, networks, meta_parameters, low_parameters, high_parameters):
    env.reset()
    terminated = False

    state = env.game.board_to_tensor(env.colour)
    state_legal_moves = env.game.board_to_legal_moves()
    state_prime_legal_moves = state_legal_moves

    high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])
    cum_r = 0

    i = 0
    while not terminated:
        legal_actions_mask = env.game.board_to_legal_moves()
        high_action = high_action_prime

        if meta_parameters["ai_low_network"]:
            action = sample_ai_low_action(env.game, high_action, meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
        else:
            action = int(sample_low_action(networks["low_network"], state, high_action, low_parameters["epsilon"], legal_actions_mask))
        (state_prime, reward, d) = env.step(action)
        low_r = get_low_r(high_action, state, state_prime)

        state_legal_moves = state_prime_legal_moves
        state_prime_legal_moves = env.game.board_to_legal_moves()
        terminated = d > 0.999

        i += 1

        action = tf.reshape(action, (1, 1))
        high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])

        cum_r += reward[0, 0]
        state = state_prime

    return cum_r

def train(environment_obj, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, inital_weights=None, evaluation_agent=Agent(), transfer_evaluation_obj=None):
    gc.collect()
    tf.keras.backend.clear_session()

    random.seed(meta_parameters["random_seed"])
    tf.random.set_seed(meta_parameters["random_seed"])

    wipe_dir(env_parameters["high_network_weights_directory"])
    wipe_dir(env_parameters["low_network_weights_directory"])
    wipe_dir(meta_parameters["weights_dir"])

    #Initialize actor and critic networks and target networks and initialize parameters
    (low_network, low_network_target, high_network, high_network_target, high_network_critic, high_network_critic_target) = initialize_networks(True, high_parameters["regularization"])

    orig_low_network_last_weights = low_network.layers[-1].get_weights()
    orig_high_network_last_weights = high_network.layers[-3].get_weights()
    orig_high_network_critic_last_weights = high_network_critic.layers[-1].get_weights()

    networks = {"low_network": low_network,
                "low_network_target": low_network_target,
                "high_network": high_network,
                "high_network_critic": high_network_critic,
                "high_network_target": high_network_target,
                "high_network_critic_target": high_network_critic_target}

    if meta_parameters["load_low_weights"]:
        load_low_weights(networks, low_parameters)
    if meta_parameters["load_high_weights"]:
        load_high_weights(networks, high_parameters)

    if meta_parameters["low_freeze_all_but_last"]:
        for layer in networks["low_network"].layers[:-1]:
            layer.trainable = False
        networks["low_network"].layers[-1].set_weights(orig_low_network_last_weights)
        networks["low_network_target"].layers[-1].set_weights(orig_low_network_last_weights)
    
    if meta_parameters["high_freeze_all_but_last"]:
        for layer in networks["high_network"].layers[:-3]:
            layer.trainable = False
        for layer in networks["high_network_critic"].layers[:-1]:
            layer.trainable = False

        networks["high_network"].layers[-3].set_weights(orig_high_network_last_weights)
        networks["high_network_critic"].layers[-1].set_weights(orig_high_network_critic_last_weights)

    #Initialize replay buffers
    low_replay_buffer = ReplayBuffer(low_parameters["replay_buffer_size"])
    high_replay_buffer = ReplayBuffer(high_parameters["replay_buffer_size"])
    val_low_replay_buffer = ReplayBuffer(low_parameters["replay_buffer_size"])
    val_high_replay_buffer = ReplayBuffer(high_parameters["replay_buffer_size"])

    replay_buffers = {"low": low_replay_buffer,
                      "high": high_replay_buffer,
                      "val_low": val_low_replay_buffer,
                      "val_high": val_high_replay_buffer}

    #Initialize optimizers
    optimizers = {
        "opt_low": tf.keras.optimizers.Adam(learning_rate=low_parameters["actor_lr"], clipvalue=low_parameters["actor_clipvalue"]),
        "opt_high": tf.keras.optimizers.Adam(learning_rate=high_parameters["actor_lr"], clipvalue=high_parameters["actor_clipvalue"]),
        "opt_high_critics": tf.keras.optimizers.Adam(learning_rate=high_parameters["critic_lr"], clipvalue=high_parameters["critic_clipvalue"]),
        "opt_high_predictor": tf.keras.optimizers.Adam(learning_rate=high_parameters["curiosity_lr"])
    }

    update_functions = {
        "update_actor_1": tf.function(update_actor_1),
        "update_critic_1": tf.function(update_critic_1),
        "update_network_2": tf.function(update_network_2),
        "update_predictor": tf.function(update_predictor)
    }

    if meta_parameters["is_learning_environment"]:
        envs = [environment_obj(env_parameters, "minimax_1"), environment_obj(env_parameters, "minimax_2"), environment_obj(env_parameters, "minimax_3")]
        val_env = environment_obj(env_parameters, "mcts_50")
    else:
        envs = [environment_obj(env_parameters)]
        val_env = environment_obj(env_parameters)
    network_N = 1

    if transfer_evaluation_obj is not None:
        transfer_env = transfer_evaluation_obj(env_parameters)
    else:
        transfer_env = None

    #Initialize tensorflow logging
    log_dir = meta_parameters["log_dir"]
    summary_writer = tf.summary.create_file_writer(log_dir)
    steps = [0 for _ in range(19)]
    start_time = time.time()
    score_history = collections.deque([])
    n_epochs_to_train = meta_parameters["n_epochs"]
    low_r_history = collections.deque()
    r_history = collections.deque()
    epoch_r_history = collections.deque()
    val_low_r_history = collections.deque()
    val_r_history = collections.deque()

    low_l = 0
    high_l = 0

    env_i = 0
    last_env_i = -1
    env_rs = [collections.deque([]) for _ in range(4)]

    misc_functions.wipe_dir(meta_parameters["evaluation_output_dir"])

    for epoch in range(1, meta_parameters["n_epochs"]+1):
        print("Epoch {}/{}".format(epoch, meta_parameters["n_epochs"]))

        if meta_parameters["is_learning_environment"]:
            env_i = choose_env_i(env_rs, meta_parameters["student_required_ma"], meta_parameters["student_required_n"], meta_parameters["student_past_agents_prob"])
            env = envs[env_i]
        else:
            env_i = 0
            env = envs[0]
        (_, cum_r, low_l, high_l) = train_1_epoch(env, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, optimizers, summary_writer, update_functions, steps, r_history, low_r_history, low_l, high_l)
        
        if meta_parameters["run_validation"]:
            validate_1_epoch(val_env, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, val_low_r_history, val_r_history, summary_writer, steps)


        last_env_i = env_i
        update_moving_averages(env_rs, last_env_i, cum_r, meta_parameters)

        if epoch % meta_parameters["latest_player_save_period"] == 0:
            network_N = save_networks(networks, network_N, meta_parameters)

        if epoch % meta_parameters["gc_period"] == 0:
            do_garbage_collection()

        if epoch >= meta_parameters["student_required_n"]:
            record_epoch_history(summary_writer, last_env_i, env_rs, steps)

        if last_env_i >= meta_parameters["training_level_cutoff"]:
            n_epochs_to_train = epoch
            break

        if (epoch - 1) % meta_parameters["evaluation_period"] == 0:
            if transfer_env is not None:
                do_transfer_evaluation(transfer_env, networks, meta_parameters, low_parameters, high_parameters)
            else:
                do_evaluation(meta_parameters, networks, high_parameters, low_parameters, evaluation_agent)

        steps[7] += 1

    save_networks(networks, network_N, meta_parameters)

    if meta_parameters["is_othello_environment"]:
        evaluation_agents = [MinimaxAgent(1)]

        s = time.time()
        print("Getting best agent...")
        (best_high_weights, best_low_weights) = get_best_agent(meta_parameters, meta_parameters["weights_dir"], meta_parameters["weights_dir"], network_N, meta_parameters["evaluation_n"], evaluation_agents, low_parameters, high_parameters)
        print("Getting best agent took {:.3f} minutes".format((time.time() - s) / 60))
        
        s = time.time()
        print("Creating results data...")
        results = create_results_HIRO(meta_parameters, meta_parameters["results_n"], best_high_weights, best_low_weights, low_parameters["epsilon"], high_parameters["epsilon"])
        print("Creating results table took {:.3f} minutes".format((time.time() - s) / 60))

        misc_functions.wipe_dir(meta_parameters["results_output_dir"])
        write_results("{}/results.csv".format(meta_parameters["results_output_dir"]), results, meta_parameters["results_n"])
    print("All training took {:.3f} minutes".format((time.time() - start_time) / 60))
    print("Training took {} epochs".format(n_epochs_to_train))
    print("All done!")

    return (n_epochs_to_train, results[("hiroAgent", "agent")], meta_parameters["results_n"])

def get_best_agent(meta_parameters, high_network_weights_dir, low_network_weights_dir, network_N, evaluation_n, evaluation_agents, low_paramaters, high_paramaters):
    best_score = float('-inf')
    best_high_weights = None
    best_low_weights = None
    if evaluation_n == 0:
        best_high_weights = "{}/high_network-{}.h5".format(high_network_weights_dir, network_N)
        best_low_weights = "{}/low_network-{}.h5".format(low_network_weights_dir, network_N)
    else:
        for i in range(1, network_N+1):
            print("Evaluating agent {}/{}".format(i, network_N))
            high_weights_path = "{}/high_network-{}.h5".format(high_network_weights_dir, i)
            low_weights_path = "{}/low_network-{}.h5".format(low_network_weights_dir, i)
            hiro_agent = HIROAgent(high_weights_path, low_weights_path, low_paramaters["epsilon"], high_paramaters["epsilon"], meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
            score = 0
            for agent in evaluation_agents:
                score += Tournament.play_n_games(evaluation_n, hiro_agent, agent, "./tmp/1", "./tmp/2")
            if score > best_score:
                best_high_weights = high_weights_path
                best_low_weights = low_weights_path
                best_score = score

    print("Chosen set of high network weights were {}".format(best_high_weights))
    print("Chosen set of low network weights were {}".format(best_low_weights))
    return (best_high_weights, best_low_weights)

@tf.function
def update_critic_1_loss(critic_network, critic_target, actor_target, s, a, r, s_prime, gamma, training=False):
    y = r + gamma * critic_target([s_prime, actor_target(s_prime)])
    L = tf.reduce_mean(tf.square(critic_network([s, a], training=training) - y))
    if training:
        L = L + tf.reduce_sum(critic_network.losses)
    return L

def update_critic_1(critic_network, critic_target, actor_target, s, a, r, s_prime, gamma, opt_critic):
    with tf.GradientTape() as tape:
        L = update_critic_1_loss(critic_network, critic_target, actor_target, s, a, r, s_prime, gamma, training=True)
    
    grads = tape.gradient(L, critic_network.trainable_variables)
    opt_critic.apply_gradients(zip(grads, critic_network.trainable_variables))

    return L

def update_predictor(network, s, a, r, s_prime, opt_predictor):
    with tf.GradientTape() as tape:
        (pred_r, pred_s_prime) = network([s, a])
        L = tf.reduce_mean(tf.square(pred_r - r)) + tf.reduce_mean(tf.square(pred_s_prime - s_prime))
    
    grads = tape.gradient(L, network.trainable_variables)
    opt_predictor.apply_gradients(zip(grads, network.trainable_variables))

    return L

@tf.function
def get_predictor_loss(network, s, a, r, s_prime):
    (pred_r, pred_s_prime) = network([s, a])
    L = tf.reduce_mean(tf.square(pred_r - r)) + tf.reduce_mean(tf.square(pred_s_prime - s_prime))
    return L

@tf.function
def update_network_2_loss(network, target_network, parameters, s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves, training=False):
    output = target_network([s_prime, g_prime])
    adjusted_output = adjusted_low_output(output, s_prime_legal_moves)
    max_a = tf.reduce_max(adjusted_output, 1)
    d = tf.reshape(d, (-1,))
    r = tf.reshape(r, (-1,))

    #Compute targets
    yi = r + (parameters["gamma"] * (1.0 - d) * max_a)
    
    #Update Q functions
    output = network([s, g], training=training)
    a = tf.reshape(tf.one_hot(a, N * N - 4, axis=1), (-1, N * N - 4))
    v = tf.reduce_sum(output * a, axis=1)
    L = tf.reduce_mean(tf.square(v - yi))
    if training:
        L = L + tf.reduce_sum(network.losses)
    return L

def update_network_2(network, target_network, parameters, s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves, opt):
    with tf.GradientTape() as tape:
        L = update_network_2_loss(network, target_network, parameters, s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves, training=True)

    grads = tape.gradient(L, network.trainable_variables)
    opt.apply_gradients(zip(grads, network.trainable_variables))

    return L

@tf.function
def update_actor_1_loss(actor_network, critic_network, s, training=False):
    a = actor_network(s, training=training)
    L = -tf.reduce_mean(critic_network([s, a]))
    if training:
        L = L + tf.reduce_sum(actor_network.losses)
    return L

def update_actor_1(actor_network, critic_network, s, opt):
    with tf.GradientTape() as tape:
        L = update_actor_1_loss(actor_network, critic_network, s, training=True)

    grads = tape.gradient(L, actor_network.trainable_variables)
    opt.apply_gradients(zip(grads, actor_network.trainable_variables))

    return L

@tf.function
def sample_high_action(high_network, state, high_epsilon, high_c, high_a_low, high_a_high, train_high):
    #Sample new goal
    if train_high:
        output = high_network(state, training=False)
        if high_epsilon > 1e-5:
            noise = output + tf.random.normal((1, GOAL_SIZE), stddev=high_epsilon)
        else:
            noise = 0.0
        high_action = output + noise
        high_action = tf.clip_by_value(high_action, -high_c, high_c)
        return high_action
    else:
        x = tf.zeros((GOAL_SIZE,)) + 1
        return x

def get_legal_actions(legal_actions_mask):
    legal_actions = [int(x[0]) for x in tf.where(legal_actions_mask[0, :])]
    return legal_actions

def sample_random_low_action(legal_actions_mask):
    return random.choice(get_legal_actions(legal_actions_mask))

@tf.function
def adjusted_low_output(output, legal_actions_mask):
    adjusted_output = output * legal_actions_mask
    adjusted_output = adjusted_output - (1 - legal_actions_mask) * (tf.reduce_max(tf.abs(output)) + 0.1)
    return adjusted_output

@tf.function
def sample_q_low_action(low_network, state, high_action, low_epsilon, legal_actions_mask):
    output = low_network([state, high_action])
    adjusted_output = adjusted_low_output(output, legal_actions_mask)

    best_action = tf.argmax(adjusted_output, axis=1)
    return best_action

def sample_low_action(low_network, state, high_action, low_epsilon, legal_actions_mask):    
    if random.random() < low_epsilon:
        best_action = sample_random_low_action(legal_actions_mask)
    else:
        best_action = sample_q_low_action(low_network, state, high_action, low_epsilon, legal_actions_mask)
        
    return best_action

def sample_ai_low_action(game, high_action, ai_low_network_depth, seed):
    goal = np.zeros((GOAL_SIZE,))
    goal[:] = high_action[0, :]
    
    move = othello.get_minimax_goal_move(game.game.board, goal, game.get_player_turn(), ai_low_network_depth, seed)
    return game.cord_to_number((move.x, move.y))

@tf.function
def obtain_goal_relabel_score(networks, s, a, la, s_legal_moves):
    current_la = tf.expand_dims(tf.argmax(adjusted_low_output(networks["low_network"]([s, a]), s_legal_moves), 1, output_type=tf.int32), 1)
    goal_relabel_score = tf.cast(tf.clip_by_value(1 - tf.abs(current_la - la), 0, 1), tf.float32)
    return goal_relabel_score

def validate_1_epoch(env, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, val_low_r_history, val_r_history, summary_writer, steps):
    env.reset()
    terminated = False

    state = env.game.board_to_tensor(env.colour)
    state_legal_moves = env.game.board_to_legal_moves()
    state_prime_legal_moves = state_legal_moves

    high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])
    cum_r = 0

    i = 0
    while not terminated:
        legal_actions_mask = env.game.board_to_legal_moves()
        high_action = high_action_prime

        if meta_parameters["ai_low_network"]:
            action = sample_ai_low_action(env.game, high_action, meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
        else:
            action = int(sample_low_action(networks["low_network"], state, high_action, low_parameters["epsilon"], legal_actions_mask))
        (state_prime, reward, d) = env.step(action)
        low_r = get_low_r(high_action, state, state_prime)

        state_legal_moves = state_prime_legal_moves
        state_prime_legal_moves = env.game.board_to_legal_moves()
        terminated = d > 0.999

        i += 1

        action = tf.reshape(action, (1, 1))
        high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])

        experience = [(state, high_action, action, reward, d, state_prime, high_action_prime, state_legal_moves, state_prime_legal_moves)]

        (s, g, a, _, d, s_prime, g_prime, _, s_prime_legal_moves) = experience[0]
        #Store transition in replay buffer
        replay_buffers["val_low"].store((s, g, a, low_r, d, s_prime, g_prime, s_prime_legal_moves))

        (s, a, la, r, _, s_prime, _, s_legal_moves, s_prime_legal_moves) = experience[0]
        replay_buffers["val_high"].store((s, a, la, r, s_prime, s_legal_moves))

        if (len(replay_buffers["val_low"]) >= low_parameters["batch_N"] or meta_parameters["ai_low_network"]) and len(replay_buffers["val_high"]) >= high_parameters["batch_N"]:
            (s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves) = replay_buffers["val_low"].sample_minibatch(low_parameters["batch_N"])
            L = update_network_2_loss(networks["low_network"], networks["low_network_target"], low_parameters, s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves)

            with summary_writer.as_default():
                tf.summary.scalar('val_low_loss', L, step=steps[18])

            (s, a, la, r, s_prime, s_legal_moves) = replay_buffers["val_high"].sample_minibatch(high_parameters["batch_N"])
        
            goal_relabel_score = obtain_goal_relabel_score(networks, s, a, la, s_legal_moves)

            if not meta_parameters["ai_low_network"]:
                (s, a, la, r, s_prime, s_legal_moves) = _relabelled_sample((s, a, la, r, s_prime, s_legal_moves), goal_relabel_score)

            critic_L = update_critic_1_loss(networks["high_network_critic"], networks["high_network_critic_target"], networks["high_network"], s, a, r, s_prime, high_parameters["gamma"])
            actor_L = update_actor_1_loss(networks["high_network"], networks["high_network_critic"], s)

            with summary_writer.as_default():
                tf.summary.scalar('val_high_critic_loss', critic_L, step=steps[18])
                tf.summary.scalar('val_high_actor_loss', actor_L, step=steps[18])

            steps[18] += 1

        val_low_r_history.append(float(low_r))
        if len(val_low_r_history) > meta_parameters["low_r_moving_average_n"]:
            val_low_r_history.popleft()
            
            with summary_writer.as_default():
                tf.summary.scalar('val_moving_average_low_reward', tf.reduce_mean(val_low_r_history), step=steps[14])
            
            steps[14] += 1

        cum_r += reward[0, 0]
        state = state_prime

    val_r_history.append(float(cum_r))
    if len(val_r_history) > meta_parameters["high_r_moving_average_n"]:
        val_r_history.popleft()

        with summary_writer.as_default():
            tf.summary.scalar('val_moving_average_epoch_reward', tf.reduce_mean(val_r_history), step=steps[15])
        
        steps[15] += 1

def train_1_epoch(env, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, optimizers, summary_writer, update_functions, steps, r_history, low_r_history, low_l, high_l):
    env.reset()
    terminated = False

    state = env.game.board_to_tensor(env.colour)
    state_legal_moves = env.game.board_to_legal_moves()
    state_prime_legal_moves = state_legal_moves

    high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])
    cum_r = 0

    i = 0
    while not terminated:
        legal_actions_mask = env.game.board_to_legal_moves()
        high_action = high_action_prime

        if meta_parameters["ai_low_network"]:
            action = sample_ai_low_action(env.game, high_action, meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
        else:
            action = int(sample_low_action(networks["low_network"], state, high_action, low_parameters["epsilon"], legal_actions_mask))
        (state_prime, reward, d) = env.step(action)
        low_r = get_low_r(high_action, state, state_prime)[0, 0]

        state_legal_moves = state_prime_legal_moves
        state_prime_legal_moves = env.game.board_to_legal_moves()
        terminated = d > 0.999

        i += 1

        action = tf.reshape(action, (1, 1))
        high_action_prime = sample_high_action(networks["high_network"], state, high_parameters["epsilon"], high_parameters["c"], high_parameters["a_low"], high_parameters["a_high"], meta_parameters["train_high"])

        experience = [(state, high_action, action, reward, d, state_prime, high_action_prime, state_legal_moves, state_prime_legal_moves)]

        cum_r += reward[0, 0]

        low_r_history.append(float(low_r))
        if len(low_r_history) > meta_parameters["low_r_moving_average_n"]:
            low_r_history.popleft()
            
            with summary_writer.as_default():
                tf.summary.scalar('moving_average_low_reward', tf.reduce_mean(low_r_history), step=steps[12])
        
            steps[12] += 1

        state = state_prime

        if meta_parameters["train_high"]:
            train_high_network(env, experience, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, optimizers, summary_writer, update_functions, steps)
        if not meta_parameters["ai_low_network"]:
            low_l = train_low_network(env, experience, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, optimizers["opt_low"], summary_writer, update_functions, steps, low_r_history, low_l)

    r_history.append(float(cum_r))
    if len(r_history) > meta_parameters["high_r_moving_average_n"]:
        r_history.popleft()

        with summary_writer.as_default():
            tf.summary.scalar('moving_average_epoch_reward', tf.reduce_mean(r_history), step=steps[16])
        
        steps[16] += 1

    return (None, cum_r, low_l, high_l)

@tf.function
def _calculate_score(values, const=1e-2):
    a = tf.cast(tf.reduce_sum(tf.maximum(values, 0.0)), dtype=tf.float32)
    b = tf.cast(tf.abs(tf.reduce_sum(tf.minimum(values, 0.0))), dtype=tf.float32)
    if a + b < const:
        return 0.0
    else:
        return a / (a + b)

@tf.function
def _get_low_r(g, board):
    stability = board_to_stability(board)
    lanes_owned = board_to_lanes_owned(board)

    coin_parity = _calculate_score(board) * g[:, 0]
    stability_score = _calculate_score(stability) * g[:, 1]
    lanes_owned = _calculate_score(lanes_owned) * g[:, 2]
    corners = _calculate_score([board[:, 0, 0, 0], board[:, 7, 0, 0], board[:, 0, 7, 0], board[:, 7, 7, 0]]) * g[:, 3]
    good_edges = _calculate_score(tf.concat([board[:, 2:6, 0, 0], board[:, 0, 2:6, 0], board[:, 2:6, 7, 0], board[:, 7, 2:6, 0]], axis=0)) * g[:, 4]

    #mobility = tf.reduce_sum(board_legal_moves) / 21.0 * g[0, 7]

    score = (coin_parity + stability_score + lanes_owned + corners + good_edges) / tf.reduce_sum(g[:, :])
    score = (score - 0.5) * 2.0
    return tf.reshape(score, (1, 1))

def get_low_r(g, s, s_prime):
    return _get_low_r(g, s_prime)

def do_low_update(replay_buffers, low_parameters, update_functions, networks, summary_writer, steps, opt_low):
    #Sample random minibatch of transitions
    (s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves) = replay_buffers["low"].sample_minibatch(low_parameters["batch_N"])

    #Update critics
    L = update_functions["update_network_2"](networks["low_network"], networks["low_network_target"], low_parameters, s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves, opt_low)

    with summary_writer.as_default():
        tf.summary.scalar('q_network_loss', L, step=steps[2])
        tf.summary.scalar('low_batch_reward', tf.reduce_mean(r), step=steps[2])

    steps[2] += 1

def board_to_lanes_owned(board):
    tensor = np.zeros((1, N, N, 1), dtype=np.float32)
    for i in range(N):
        if tf.reduce_sum(tf.abs(board[0, i, :, 0])) == N:
            tensor[0, i, :, 0] += 1.0
        if tf.reduce_sum(tf.abs(board[0, :, i, 0])) == N:
            tensor[0, :, i, 0] += 1.0
        
    for i in range(N):
        for j in range(N):
            tensor[0, i, j, 0] += _n_diagonals_filled(board, i, j)

    tensor[0, 0, 0, 0] = 4
    tensor[0, 7, 7, 0] = 4
    tensor[0, 0, 7, 0] = 4
    tensor[0, 7, 0, 0] = 4
    
    return tensor * board

def board_to_stability(board):
    lanes_owned = board_to_lanes_owned(board)
    return tf.cast(board_to_lanes_owned(board) == 4, dtype=tf.float32) - tf.cast(board_to_lanes_owned(board) == -4, dtype=tf.float32)

def _n_diagonals_filled(board, i, j):
    score1 = 0
    score2 = 0

    for (di, dj) in [(1, 1), (-1, -1)]:
        score1 += 1 if _is_diagonal_filled(board, i, j, di, dj) else 0
    score1 = 1 if score1 > 1 else 0

    for (di, dj) in [(1, -1), (-1, 1)]:
        score2 += 1 if _is_diagonal_filled(board, i, j, di, dj) else 0
    score2 = 1 if score2 > 1 else 0

    return score1 + score2

def _is_diagonal_filled(board, i, j, di, dj):
    l = []
    while i < N and j < 0 and i >= 0 and j >= 0:
        l.append(tf.abs(board[0, i, j, 0]))
        i += di
        j += dj
    if l == 0:
        filled = True
    else:
        filled = round(sum(l)) == len(l)
    return filled

def train_low_network(env, experience, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, opt_low, summary_writer, update_functions, steps, low_r_history, l):
    #Observe state and select action
    #Execute action(state, high_output, high_action, action, reward, d, state_prime, high_output_prime, high_action_prime, state_prime_legal_moves) = experience[0]
    (s, g, a, _, d, s_prime, g_prime, _, s_prime_legal_moves) = experience[0]
    
    r = get_low_r(g, s, s_prime)
    
    #Store transition in replay buffer
    replay_buffers["low"].store((s, g, a, r, d, s_prime, g_prime, s_prime_legal_moves))

    for k in range(low_parameters["n_updates"]):
        #If s' terminal, reset environment
        #If time to update
        if len(replay_buffers["low"]) >= low_parameters["batch_N"] and len(replay_buffers["high"]) >= high_parameters["batch_N"] and l % low_parameters["update_period"] == 0:
            do_low_update(replay_buffers, low_parameters, update_functions, networks, summary_writer, steps, opt_low)
            update_weights(networks["low_network_target"].variables, networks["low_network"].variables, low_parameters["rho"])
        l += 1

    return l

def train_high_network(env, experience, networks, replay_buffers, meta_parameters, low_parameters, high_parameters, optimzers, summary_writer, update_functions, steps):
    (s, a, la, r, _, s_prime, _, s_legal_moves, s_prime_legal_moves) = experience[0]

    replay_buffers["high"].store((s, a, la, r, s_prime, s_legal_moves))
    
    if (len(replay_buffers["low"]) >= low_parameters["batch_N"] or meta_parameters["ai_low_network"]) and len(replay_buffers["high"]) >= high_parameters["batch_N"]:
        (s, a, la, r, s_prime, s_legal_moves) = replay_buffers["high"].sample_minibatch(high_parameters["batch_N"])
        
        relabel_goals = not meta_parameters["ai_low_network"] and not meta_parameters["disable_goal_relabelling"]

        if relabel_goals:
            goal_relabel_score = obtain_goal_relabel_score(networks, s, a, la, s_legal_moves)
            (s, a, la, r, s_prime, s_legal_moves) = _relabelled_sample((s, a, la, r, s_prime, s_legal_moves), goal_relabel_score)
        else:
            goal_relabel_score = [1]

        L = update_functions["update_critic_1"](networks["high_network_critic"], networks["high_network_critic_target"], networks["high_network_target"], s, a, r, s_prime, high_parameters["gamma"], optimzers["opt_high_critics"])
        
        with summary_writer.as_default():
            tf.summary.scalar('high_critic_loss', L, step=steps[13])
            tf.summary.scalar('goal_relabel_score', tf.reduce_mean(goal_relabel_score), step=steps[13])
            
        L = update_functions["update_actor_1"](networks["high_network"], networks["high_network_critic"], s, optimzers["opt_high"])

        with summary_writer.as_default():
            tf.summary.scalar('high_actor_loss', L, step=steps[13])
        
        steps[13] += 1

        update_weights(networks["high_network_target"].variables, networks["high_network"].variables, high_parameters["rho"])
        update_weights(networks["high_network_critic_target"].variables, networks["high_network_critic"].variables, high_parameters["rho"])

def _relabelled_sample(transitions, goal_relabel_score):
    (s, a, la, r, s_prime, s_legal_moves) = transitions

    batch_score = tf.reduce_sum(goal_relabel_score)

    if batch_score < len(transitions):
        indices = tf.where(goal_relabel_score == 1)
        indices = tf.concat([tf.random.uniform((len(transitions[0]) - batch_score,), minval=0, maxval=batch_score), tf.range(batch_score)], 0)
        ret = tuple([tf.gather(x, indices) for x in transitions])
    else:
        ret = transitions

    assert len(ret) == len(transitions)
    return ret

@tf.function
def update_weights(target_network_weights: tf.keras.Model, network_weights: tf.keras.Model, rho: float):
    for (target_weights, weights) in zip(target_network_weights, network_weights):
        target_weights.assign(weights * rho + (1 - rho) * target_weights)