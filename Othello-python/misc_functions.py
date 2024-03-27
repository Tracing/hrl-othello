from constants import N
import math
import os
import numpy as np
import random
import tensorflow as tf

def choose_env_i(env_rs, required_ma, required_n, prev_opponent_prob):
    i = 0
    while i < 3 and (len(env_rs[i]) >= required_n) and np.mean(env_rs[i]) > required_ma:
        i += 1
    
    if random.random() < prev_opponent_prob and i > 0:
        ret = random.randint(0, i)
    else:
        ret = i

    return ret

@tf.function
def normal_pdf(x, u, sigma):
    return (1 / (sigma * tf.sqrt(2 * math.pi))) * tf.exp(-0.5 * tf.square((x - u) / sigma))

@tf.function
def normal_pdf_log_likelihood(x, u, sigma):
    return -0.5 * tf.math.log(2 * math.pi) - 0.5 * tf.math.log(sigma ** 2) - (1 / (2 * sigma ** 2)) * (x - u) ** 2

@tf.function
def multivariate_normal_pdf(x: tf.Tensor, u: tf.Tensor, sigma: float):
    x = tf.reshape(x, (-1, N * N))
    u = tf.reshape(x, (-1, N * N))
    k = x.shape[1]
    c = tf.eye(x.shape[1]) * sigma
    return (2 * math.pi) ** (- k / 2) * tf.linalg.det(c) ** (-0.5) * tf.exp(-0.5 * tf.matmul(tf.matmul((x - u), c), tf.transpose(x - u)))

def log_multivariate_normal_pdf(x: tf.Tensor, u: tf.Tensor, sigma: float):
    x = tf.reshape(x, (-1, N * N))
    u = tf.reshape(x, (-1, N * N))
    k = x.shape[1]
    m = tf.eye(x.shape[1]) * sigma
    a = tf.math.log((2 * math.pi) ** (- k / 2))
    b = tf.linalg.det(m) ** (-0.5)
    c = -0.5 * tf.matmul(tf.matmul((x - u), m), tf.transpose(x - u))
    return (a + b + c)[0, 0]

@tf.function
def multivariate_normal_log_likelihood(x: tf.Tensor, u: tf.Tensor, sigma: float):
    x = tf.reshape(x, (-1, N * N))
    u = tf.reshape(x, (-1, N * N))
    k = x.shape[1]
    return (-0.5 * tf.reduce_sum(((x - u) ** 2) / (sigma ** 2)) + 2 * k * tf.math.log(sigma + 1e-5) + k * 2 * 1.837877)

@tf.function
def sample_action(state, network, sigma):
    output = network(state)
    noise = tf.random.normal(output.shape, stddev=sigma)
    action = output + noise
    return action

def wipe_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for file_path in os.listdir(path):
        os.remove("{}/{}".format(path, file_path))

@tf.function
def state_to_high_action(state):
    return tf.reshape(state[:, :, :, 0], (1, N, N, 1))
