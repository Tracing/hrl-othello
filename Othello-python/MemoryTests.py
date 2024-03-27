from constants import WHITE, N
from ddpg import ReplayBuffer
import Environments
import gc
import othello
import othello_wrapper
import random
import tensorflow as tf

def test_board_to_tensor():
    game = othello_wrapper.OthelloGame()
    a = game.board_to_tensor(WHITE)
    input(">>")
    for i in range(10000):
        a = game.board_to_tensor(WHITE)
    input(">>") 

def test_game_playthrough():
    random.seed(1)

    n = 100
    game = othello_wrapper.OthelloGame()
    input(">>")
    for i in range(n):
        game.reset()
        while not game.game_has_ended():
            game.make_move_2(random.choice(game.get_moves_2()))
        gc.collect()
    input(">>")

def test_ddpg_replaybuffer():
    s = 4096
    n = 10000
    M = 10000
    data = []
    for _ in range(n):
        data.append((tf.random.normal((1, N, N, 3)), tf.random.normal((1, N, N, 1)), 0, tf.random.normal((1, N, N, 3))))
    
    buffer = ReplayBuffer(s)
    for i in range(s):
        buffer.store(data[i])
    input(">>")
    for i in range(s, n):
        buffer.store(data[i])
    for m in range(M):
        a = buffer.sample_minibatch(1000)
    gc.collect()
    input(">>")

def test_game_get_moves_2():
    n = 1000

    game = othello_wrapper.OthelloGame()
    input(">>")
    for i in range(n):
        moves = game.get_moves_2()
    gc.collect()
    input(">>")

def test_environment_reset():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 1000
   
    environment = Environments.OthelloEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
    input(">> ")

def test_environment_step():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 1000
   
    environment = Environments.OthelloEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
        action = tf.random.normal((1, N, N, 1))
        environment.step(action)
        gc.collect()
    input(">> ")

def test_environment_run():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 100
   
    environment = Environments.OthelloEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
        d = 0.0
        while d < 0.999:
            action = tf.random.normal((1, N, N, 1))
            (_, _, d) = environment.step(action)
        gc.collect()
    input(">> ")

def test_learning_environment_reset():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 1000
   
    environment = Environments.OthelloLearningEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
    input(">> ")

def test_learning_environment_step():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 1000
   
    environment = Environments.OthelloLearningEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
        action = tf.random.normal((1, N, N, 1))
        environment.step(action)
        gc.collect()
    input(">> ")

def test_learning_environment_run():
    env_parameters = {
    "high_network_weights_directory": "HIRO_high_weights",
    "low_network_weights_directory": "HIRO_low_weights",
    "h": 1,
    "previous_opponent_prob": 0.2,
    "max_random_moves_before_game": 10,
    "epsilon": 0.00
    }
    n = 100
   
    environment = Environments.OthelloLearningEnvironment(env_parameters)
    input(">> ")
    for i in range(n):
        environment.reset()
        d = 0.0
        while d < 0.999:
            action = tf.random.normal((1, N, N, 1))
            (_, _, d) = environment.step(action)
        gc.collect()
    input(">> ")

if __name__ == "__main__":
    print("Testing environment reset...")
    #test_environment_reset()
    print("Testing environment step...")
    #test_environment_step()
    print("Testing environment run...")
    #test_environment_run()
    print("Testing learning environment reset...")
    #test_learning_environment_reset()
    print("Testing learning environment step...")
    #test_learning_environment_step()
    print("Testing learning environment run...")
    #test_learning_environment_run()
    print("Testing game playthrough...")
    #test_game_playthrough()
    print("Testing get_moves_2...")
    #test_game_get_moves_2()
    print("Testing ddpg replay buffer...")
    test_ddpg_replaybuffer()