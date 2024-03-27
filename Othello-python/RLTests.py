import Environments as envs
import ddpg
import td3
import tensorflow as tf
import HIRO

def get_ddpg_network_f(state_dim, verbose=False):
    def f(verbose=False):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = tf.keras.layers.Input(state_dim)
        x = tf.keras.layers.Dense(256, activation="relu")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=last_init)(x)
        outputs = outputs * 2.0
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if verbose:
            model.summary()
        return model
    return f

def get_ddpg_critic_f(state_dim, action_dim, verbose=False):
    def f(verbose=False):
        inputs1 = tf.keras.layers.Input(state_dim)
        x1 = tf.keras.layers.Dense(16, activation="relu")(inputs1)
        x1 = tf.keras.layers.Dense(32, activation="relu")(x1)

        inputs2 = tf.keras.layers.Input(action_dim)
        x2 = tf.keras.layers.Dense(32, activation="relu")(inputs2)

        x = tf.keras.layers.Concatenate()([x1, x2])
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)

        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
        if verbose:
            model.summary()
        return model
    return f

def get_td3_network_f(state_dim, verbose=False):
    def f(verbose=False):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = tf.keras.layers.Input(state_dim)
        x = tf.keras.layers.Dense(256, activation="relu")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=last_init)(x)
        outputs = outputs * 2.0
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if verbose:
            model.summary()
        return model
    return f

def get_td3_critic_f(state_dim, action_dim, verbose=False):
    def f(verbose=False):
        inputs1 = tf.keras.layers.Input(state_dim)
        x1 = tf.keras.layers.Dense(16, activation="relu")(inputs1)
        x1 = tf.keras.layers.Dense(32, activation="relu")(x1)

        inputs2 = tf.keras.layers.Input(action_dim)
        x2 = tf.keras.layers.Dense(32, activation="relu")(inputs2)

        x = tf.keras.layers.Concatenate()([x1, x2])
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
        if verbose:
            model.summary()
        return model
    return f

def do_halfcheetah(seed=1):
    env_parameters = {
        "render_mode": None,
        "seed": seed
    }
    ddpg_parameters = {
        "n_epochs": 1000000,
        "log_dir": "logs/DDPG_HalfCheetah",
        "save_period": 1000000, 
        "batch_N": 64, 
        "replay_buffer_size": 200000, 
        "gamma": 0.99, 
        "sigma": 0.2,
        "tau": 10e-3,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "random_seed": 1,
        "weights_dir": "ddpg_halfcheetah_weights",
        "evaluation_n": 200,
        "results_n": 1000,
        "results_output_dir": "evaluation_logs_ddpg",
        "l2": 0.0,
        "critic_clipvalue": 100000.0,
        "actor_clipvalue": 100000.0,
        "reward_ma": 10,
        "cutoff_ma": 1e10,
        "update_period": 1,
        "is_othello_environment": False
    }
    env = envs.HalfCheetahEnv(env_parameters)
    ddpg.ddpg(env, ddpg_parameters, get_ddpg_network_f((17,)), get_ddpg_critic_f((17,), (6,)), tf.zeros((1, 17)), [tf.zeros((1, 17)), tf.zeros((1, 6))])

    td3_parameters = {
        "n_epochs": 1000000,
        "log_dir": "logs/TD3_HalfCheetah",
        "save_period": 1000000, 
        "batch_N": 100, 
        "replay_buffer_size": 200000,
        "gamma": 0.99,
        "epsilon": 0.1,
        "rho": 50e-3,
        "a_low": -100000,
        "a_high": 100000,
        "c": 10000,
        "update_period": 1,
        "n_updates": 1,
        "policy_delay": 1,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "random_seed": 1,
        "weights_dir": "td3_halfcheetah_weights",
        "evaluation_n": 200,
        "results_n": 1000,
        "results_output_dir": "evaluation_logs_TD3",
        "l2": 0.0,
        "critic_clipvalue": 100000.0,
        "actor_clipvalue": 100000.0,
        "reward_ma": 10,
        "cutoff_ma": 1e10,
        "is_othello_environment": False
    }
    env = envs.HalfCheetahEnv(env_parameters)
    td3.td3(env, td3_parameters, get_td3_network_f((17,)), get_td3_critic_f((17,), (6,)), tf.zeros((1, 17)), [tf.zeros((1, 17)), tf.zeros((1, 6))])

def do_pendulum(seed=1):
    env_parameters = {
        "render_mode": None,
        "seed": seed
    }
    
    ddpg_parameters = {
        "n_epochs": 100,
        "log_dir": "logs/DDPG_Pendulum",
        "save_period": 1000000, 
        "batch_N": 64, 
        "replay_buffer_size": 50000, 
        "gamma": 0.99, 
        "sigma": 0.2,
        "damping": 0.15,
        "tau": 0.005,
        "actor_lr": 0.001,
        "critic_lr": 0.002,
        "random_seed": seed,
        "weights_dir": "ddpg_weights",
        "evaluation_n": 200,
        "results_n": 1000,
        "results_output_dir": "evaluation_logs_ddpg",
        "l2": 0.0,
        "critic_clipvalue": None,
        "actor_clipvalue": None,
        "reward_ma": 10,
        "cutoff_ma": 1e10,
        "update_period": 1,
        "is_othello_environment": False,
        "gc_period": 1000,
        "a_low": -2,
        "a_high": 2
    }
    env = envs.PendulumEnv(env_parameters)
    ddpg.ddpg(env, ddpg_parameters, get_ddpg_network_f((3,)), get_ddpg_critic_f((3,), (1,)), tf.zeros((1, 3)), [tf.zeros((1, 3)), tf.zeros((1, 1))])

    td3_parameters = {
        "n_epochs": 100,
        "log_dir": "logs/TD3_Pendulum",
        "save_period": 1000000, 
        "batch_N": 64, 
        "replay_buffer_size": 50000,
        "gamma": 0.99,
        "epsilon": 0.2,
        "damping": 0.15,
        "rho": 0.005,
        "a_low": -2,
        "a_high": 2,
        "c": 4,
        "update_period": 1,
        "n_updates": 1,
        "policy_delay": 1,
        "actor_lr": 0.001,
        "critic_lr": 0.002,
        "random_seed": seed,
        "weights_dir": "td3_weights",
        "evaluation_n": 200,
        "results_n": 1000,
        "results_output_dir": "evaluation_logs_TD3",
        "l2": 0.0,
        "critic_clipvalue": 100000.0,
        "actor_clipvalue": 100000.0,
        "reward_ma": 10,
        "cutoff_ma": 1e10,
        "is_othello_environment": False,
        "gc_period": 1000
    }
    env = envs.PendulumEnv(env_parameters)
    td3.td3(env, td3_parameters, get_td3_network_f((3,)), get_td3_critic_f((3,), (1,)), tf.zeros((1, 3)), [tf.zeros((1, 3)), tf.zeros((1, 1))])

if __name__ == "__main__":
    do_pendulum(2)