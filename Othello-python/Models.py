from constants import N, BATCH_SIZE, GOAL_SIZE
import tensorflow as tf

def initialize_networks(verbose=False, reg=0.1):
    low_network = get_low_network(verbose)
    low_network_target = get_low_network()

    high_network = get_high_network(verbose, reg)
    high_network_target = get_high_network(reg=reg)
    high_network_critic = get_high_network_critic(verbose)
    high_network_critic_target = get_high_network_critic()
    

    dummy_input = tf.zeros((BATCH_SIZE, N, N, 1))
    dummy_input2 = tf.zeros((BATCH_SIZE, GOAL_SIZE))
    dummy_input3 = tf.zeros((BATCH_SIZE, N * N - 4))

    low_network([dummy_input, dummy_input2])
    low_network_target([dummy_input, dummy_input2])

    high_network(dummy_input)
    high_network_target(dummy_input)

    high_network_critic([dummy_input, dummy_input2])
    high_network_critic_target([dummy_input, dummy_input2])

    low_network_target.set_weights(low_network.get_weights())
    high_network_target.set_weights(high_network.get_weights())
    high_network_critic_target.set_weights(high_network_critic.get_weights())

    return (low_network, low_network_target, high_network, high_network_target, high_network_critic, high_network_critic_target)

def get_dqn_network(verbose=False, reg=0.00):
    inputs = tf.keras.layers.Input((N, N, 1))
    
    x = tf.keras.layers.Reshape((-1,))(inputs)
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(N * N - 4, activation="linear")(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    if verbose:
        model.summary()
    return model

def get_high_network(verbose=False, reg=0.1):
    init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = tf.keras.layers.Input((N, N, 1))
    x = tf.keras.layers.Reshape((-1,))(inputs)
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(GOAL_SIZE, activation="sigmoid", kernel_initializer=init)(x)

    outputs = x * 0.9 + 0.1
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    if verbose:
        model.summary()
    return model

def get_high_network_critic(verbose=False, reg=0.000):
    inputs1 = tf.keras.layers.Input((N, N, 1))
    inputs2 = tf.keras.layers.Input((GOAL_SIZE,))

    x2 = inputs2

    x = tf.keras.layers.Reshape((-1,))(inputs1)
    x = tf.keras.layers.Concatenate()([x, x2])
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[outputs])
    if verbose:
        model.summary()
    return model

def get_low_network(verbose=False, reg=0.000):
    init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs1 = tf.keras.layers.Input((N, N, 1))
    inputs2 = tf.keras.layers.Input((GOAL_SIZE,))

    x = inputs1
    x2 = inputs2

    x = tf.keras.layers.Reshape((-1,))(inputs1)
    x = tf.keras.layers.Concatenate()([x, x2])
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l1_l2(reg))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(N * N - 4, activation="linear")(x)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=[outputs])
    if verbose:
        model.summary()
    return model

