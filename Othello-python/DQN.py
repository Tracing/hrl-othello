from Agents import Agent, DQN_Agent
from constants import N, NULL, BATCH_SIZE
from ReplayBuffer import ReplayBuffer
from ResultsTableProduction import get_best_agent_dqn, create_results_DQN, write_results
import collections
import gc
import math
import misc_functions
import numpy as np
import random
import tensorflow as tf
import time
import Tournament

@tf.function
def loss(network, target_network, s, a, r, s_prime, d, s_prime_legal_moves, gamma):
    output = target_network(s_prime)
    adjusted_output = output - (1 - s_prime_legal_moves) * (tf.reduce_max(tf.abs(output)) + 0.1)
    max_a = tf.reduce_max(adjusted_output, 1)
    d = tf.reshape(d, (-1,))
    r = tf.reshape(r, (-1,))

    #Compute targets
    yi = r + (gamma * (1.0 - d) * max_a)
    
    #Update Q functions
    output = network(s, training=True)
    a = tf.reshape(tf.one_hot(a, N * N - 4, axis=1), (-1, N * N - 4))
    v = tf.reduce_sum(output * a, axis=1)
    L = tf.reduce_mean(tf.square(v - yi))
    return L

def update(network, target_network, s, a, r, s_prime, d, s_prime_legal_moves, gamma, opt):
    with tf.GradientTape() as tape:
        L = loss(network, target_network, s, a, r, s_prime, d, s_prime_legal_moves, gamma)

    grads = tape.gradient(L, network.trainable_variables)
    opt.apply_gradients(zip(grads, network.trainable_variables))

    return L

def select_action(s, legal_numbers, network, epsilon):
    legal_numbers = [int(i[0]) for i in list(tf.where(legal_numbers[0, :] == 1))]

    if random.random() < epsilon:
        return tf.reshape(random.choice(legal_numbers), (1, 1))
    else:
        output = network(s)
        best_score = float('-inf')
        best_number = 0

        for i in legal_numbers:
            score = output[0, i]
            if score > best_score:
                best_score = score
                best_number = i
        return tf.reshape(best_number, (1, 1))
    return a

def initialize_dqn_networks(get_dqn_network, network_input):
    network = get_dqn_network()
    network_target = get_dqn_network()

    network(network_input)
    network_target(network_input)

    network_target.set_weights(network.get_weights())
    
    return (network, network_target)

def initiailize_optimizer(parameters):
    opt = tf.keras.optimizers.Adam(learning_rate=parameters["lr"], clipvalue=parameters["clipvalue"])

    return opt

def initialize_replay_buffers(parameters):
    return (ReplayBuffer(parameters["replay_buffer_size"]), ReplayBuffer(parameters["replay_buffer_size"]))

def initialize_tf_functions():
    update_f = tf.function(update)
    return update_f

def initialize_tf_logging_variables(parameters):
    summary_writer = tf.summary.create_file_writer(parameters["log_dir"])
    steps = [0, 0, 0, 0, 0]
    return (summary_writer, steps)

def initiailize_misc_variables(parameters):
    score_history = collections.deque([])
    val_score_history = collections.deque([])

    network_N = 1
    start_time = time.time()
    n_epochs_to_train = parameters["n_epochs"]

    return (score_history, val_score_history, network_N, start_time, n_epochs_to_train)

def validate_1_epoch(environment, parameters, replay_buffer, update_f, network, target_network, epsilon, summary_writer, steps, val_score_history):
    #Recieve initial observation state
    environment.reset()
    s = environment.state
    s_legal_moves = environment.game.board_to_legal_moves()
    terminal = False
    cum_reward = 0
    while not terminal:
        #Select action
        a = select_action(s, s_legal_moves, network, epsilon)
        #Execute action
        (s_prime, r, d) = environment.step(int(a))
        s_prime_legal_moves = environment.game.board_to_legal_moves()
        cum_reward += r[0, 0]
        terminal = d > 0.999

        #Store transition in replay buffer
        replay_buffer.store((s, a, r, s_prime, d, s_prime_legal_moves))
        if len(replay_buffer) >= parameters["batch_N"]:
            #Sample random minibatch of transitions
            (_s, _a, _r, _s_prime, _d, _s_prime_legal_moves) = replay_buffer.sample_minibatch(parameters["batch_N"])
            
            #Update critic
            L = loss(network, target_network, _s, _a, _r, _s_prime, _d, _s_prime_legal_moves, parameters["gamma"])

            with summary_writer.as_default():
                tf.summary.scalar('val_loss', L, step=steps[0])
            
            steps[2] += 1
        s = s_prime
        s_legal_moves = s_prime_legal_moves

    val_score_history.append(cum_reward)

    if len(val_score_history) > parameters["score_ma"]:
        with summary_writer.as_default():
            tf.summary.scalar('val_moving_average_epoch_reward', np.mean(val_score_history), step=steps[3])
        steps[3] += 1

        val_score_history.popleft()

def train_1_epoch(environment, parameters, replay_buffer, update_f, network, target_network, opt, rho, epsilon, summary_writer, steps):
    #Recieve initial observation state
    environment.reset()
    s = environment.state
    s_legal_moves = environment.game.board_to_legal_moves()
    terminal = False
    cum_reward = 0
    while not terminal:
        #Select action
        a = select_action(s, s_legal_moves, network, epsilon)
        #Execute action
        (s_prime, r, d) = environment.step(int(a))
        s_prime_legal_moves = environment.game.board_to_legal_moves()
        cum_reward += r[0, 0]
        terminal = d > 0.999

        #Store transition in replay buffer
        replay_buffer.store((s, a, r, s_prime, d, s_prime_legal_moves))
        if len(replay_buffer) >= parameters["batch_N"]:
            #Sample random minibatch of transitions
            (_s, _a, _r, _s_prime, _d, _s_prime_legal_moves) = replay_buffer.sample_minibatch(parameters["batch_N"])
            
            #Update critic
            L = update_f(network, target_network, _s, _a, _r, _s_prime, _d, _s_prime_legal_moves, parameters["gamma"], opt)

            with summary_writer.as_default():
                tf.summary.scalar('loss', L, step=steps[0])
            
            #Update target networks
            update_weights(target_network.variables, network.variables, rho)
            steps[0] += 1
        s = s_prime
        s_legal_moves = s_prime_legal_moves

    with summary_writer.as_default():
        tf.summary.scalar('reward', cum_reward, step=steps[1])
    steps[1] += 1

    return cum_reward

def post_training_processing(parameters, actor_network, network_N, start_time, n_epochs_to_train):
    if parameters["is_othello_environment"]:
        s = time.time()
        actor_network.save_weights("{}/dqn-{}.h5".format(parameters["weights_dir"], network_N))
        best_weights = get_best_agent_dqn(parameters["weights_dir"], network_N, parameters["evaluation_n"], [Agent()])
        
        print("Getting best agent took {:.3f} minutes".format((time.time() - s) / 60))

        s = time.time()
        results = create_results_DQN(parameters["results_n"], best_weights, parameters["results_output_dir"])

        misc_functions.wipe_dir(parameters["results_output_dir"])
        write_results("{}/results.csv".format(parameters["results_output_dir"]), results, parameters["results_n"])
        print("Creating results took {:.3f} minutes".format((time.time() - s) / 60))

    print("All training took {:.3f} minutes".format((time.time() - start_time) / 60))
    print("Training took {} epochs".format(n_epochs_to_train))
    print("All done!")

    if parameters["is_othello_environment"]:
        return (n_epochs_to_train, results[("dqnAgent", "agent")], parameters["results_n"])
    else:
        return n_epochs_to_train

def record_epoch_history(summary_writer, last_env_i, env_rs, steps):
    with summary_writer.as_default():
        tf.summary.scalar('training_level', last_env_i, step=steps[4])
        tf.summary.scalar('last_env_mean_reward', np.mean(env_rs[last_env_i]), step=steps[4])

    steps[4] += 1

def save_weights(i, parameters, network, network_N):
    if i % parameters["save_period"] == 0:
        network.save_weights("{}/dqn-{}.h5".format(parameters["weights_dir"], network_N))
        network_N += 1
    return network_N

def garbage_collection(i, parameters):
    if i % parameters["gc_period"] == 0:
        gc.collect()

def initiailize(parameters):
    tf.keras.backend.clear_session()
    gc.collect()

    random.seed(parameters["random_seed"])
    tf.random.set_seed(parameters["random_seed"])

    misc_functions.wipe_dir(parameters["weights_dir"])

def initialize_environments(environment_obj, env_parameters):
    envs = [environment_obj(env_parameters, "minimax_1"), environment_obj(env_parameters, "minimax_2"), environment_obj(env_parameters, "minimax_3")]
    val_env = environment_obj(env_parameters, "mcts_50")
    env_rs = [collections.deque([]) for _ in range(4)]
    return (envs, val_env, env_rs)

def do_evaluation(parameters, network):
    network.save_weights("./network_weights.h5")
    dqn_agent = DQN_Agent(parameters["epsilon"], "./network_weights.h5")
    agent = Agent()
    score = Tournament.play_n_games(parameters["opponent_evaluation_games"], dqn_agent, agent, "./tmp/1", "./tmp/2")

    time.sleep(1)

    with open("{}/data.csv".format(parameters["evaluation_output_dir"]), "a") as f:
        f.write("{:.2f} ".format(score))

    time.sleep(1)

def dqn(environment_obj, env_parameters, parameters, get_dqn_network, network_input=tf.zeros((BATCH_SIZE, N, N, 1))):
    initiailize(parameters)

    #Randomly initialize critic, actor and target networks
    (network, target_network) = initialize_dqn_networks(get_dqn_network, network_input)
    opt = initiailize_optimizer(parameters)
    update_f = initialize_tf_functions()
    (replay_buffer, val_replay_buffer) = initialize_replay_buffers(parameters)
    (summary_writer, steps) = initialize_tf_logging_variables(parameters)
    (score_history, val_score_history, network_N, start_time, n_epochs_to_train) = initiailize_misc_variables(parameters)
    (envs, val_env, env_rs) = initialize_environments(environment_obj, env_parameters)
    n_epochs_to_train = parameters["n_epochs"]

    misc_functions.wipe_dir(parameters["evaluation_output_dir"])

    for i in range(1, parameters["n_epochs"]+1):
        env_i = misc_functions.choose_env_i(env_rs, parameters["student_required_ma"], parameters["student_required_n"], parameters["student_past_agents_prob"])

        cum_r = train_1_epoch(envs[env_i], parameters, replay_buffer, update_f, network, target_network, opt, parameters["rho"], parameters["epsilon"], summary_writer, steps)

        if parameters["run_validation"]:
            validate_1_epoch(val_env, parameters, val_replay_buffer, update_f, network, target_network, parameters["epsilon"], summary_writer, steps, val_score_history)
        
        update_moving_averages(env_rs, env_i, cum_r, parameters)

        if i > parameters["student_required_n"]:
            record_epoch_history(summary_writer, env_i, env_rs, steps)

        if env_i >= parameters["training_level_cutoff"]:
            n_epochs_to_train = i
            break

        if i % parameters["save_period"] == 0:
            network_N = save_weights(i, parameters, network, network_N)

        if (i - 1) % parameters["evaluation_period"] == 0:
            do_evaluation(parameters, network)

        print("Epoch {}/{}".format(i, parameters["n_epochs"]))

    return post_training_processing(parameters, network, network_N, start_time, n_epochs_to_train)

def update_moving_averages(env_rs, last_env_i, cum_r, parameters):
    env_rs[last_env_i].append(cum_r)
    if len(env_rs[last_env_i]) > parameters["student_required_n"]:
        env_rs[last_env_i].popleft()

@tf.function
def update_weights(target_network_weights: tf.keras.Model, network_weights: tf.keras.Model, tau: float):
    for (x1, x2) in zip(target_network_weights, network_weights):
        x1.assign(x2 * tau + x1 * (1 - tau))
