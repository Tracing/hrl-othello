from Agents import MinimaxAgent
from Models import initialize_networks
from HIRO import train
from Environments import Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange, Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation
import sys
import tensorflow as tf

def TransferTrain(env, validation_env, name, epochs, train_low, seed=1):
    low_actor_lr = 1e-4 if train_low else 0

    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 1,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": low_actor_lr,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 1,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-3,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": True,
        "load_high_weights": True,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": False,
        "disable_goal_relabelling": not train_low
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def TransferTrainHighScratch(env, validation_env, name, epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 10000000,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 0.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 1,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-3,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": True,
        "load_high_weights": False,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": False,
        "disable_goal_relabelling": True
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def TrainFromScratch(env, validation_env, name, epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 1,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 1e-4,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 1,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-3,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": False,
        "load_high_weights": False,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": False,
        "disable_goal_relabelling": False
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def ZeroShot(env, validation_env, name, epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 10000000,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 10000000,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 0,
        "actor_lr": 0,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": True,
        "load_high_weights": True,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": False,
        "disable_goal_relabelling": True
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def TransferLinearFreeze(env, validation_env, name, epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 1,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 1e-4,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 1,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-3,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": True,
        "load_high_weights": True,
        "low_freeze_all_but_last": True,
        "high_freeze_all_but_last": True,
        "disable_goal_relabelling": False
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def TransferLinearFreezeTrainHighOnly(env, validation_env, name, epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 10000000,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/low_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network_target-9.h5"],
        "rho": 1e-4,
        "l2": 0.0
    }
    high_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 256,
        "rho": 1e-2,
        "update_period": 1,
        "n_updates": 1,
        "epsilon": 0.2,
        "gamma": 0.99,
        "c": 1000.00,
        "a_low": 0,
        "a_high": 1,
        "n_goals": 1,
        "goal_variation": 0.1,
        "critic_lr": 1e-4,
        "actor_lr": 1e-3,
        "curiosity_lr": 1e-7,
        "curiosity": 0.00,
        "critic_clipvalue": 1.0,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_target-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic-9.h5", "drive/MyDrive/othello/HIRO_weights/high_network_critic_target-9.h5"],
        "regularization": 0.1
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 10000,
        "evaluation_period": 100,
        "opponent_evaluation_games": 200,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 0,
        "evaluation_n": 0,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "student_required_n": 100,
        "student_required_ma": 0.7,
        "low_r_moving_average_n": 400,
        "high_r_moving_average_n": 400,
        "student_past_agents_prob": 0.2,
        "ai_low_network": False,
        "train_high": True,
        "training_level_cutoff": 4,
        "run_validation": False,
        "is_learning_environment": False,
        "load_low_weights": True,
        "load_high_weights": True,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": True,
        "disable_goal_relabelling": True
    }
    return train(env, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks, MinimaxAgent(1), transfer_evaluation_obj=validation_env)

def train_transfer_high_scratch(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-HighFromScratch-{}".format(env_name, seed)
        TransferTrainHighScratch(env, val_env, name, epochs, seed)

def train_transfer_from_scratch(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-FromScratch-{}".format(env_name, seed)
        TrainFromScratch(env, val_env, name, epochs, seed)

def train_zero_shot(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-ZeroShot-{}".format(env_name, seed)
        ZeroShot(env, val_env, name, epochs, seed)

def train_linear_freeze_shot(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-LinearFreeze-{}".format(env_name, seed)
        TransferLinearFreeze(env, val_env, name, epochs, seed)

def train_linear_freeze_shot_high_only(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-LinearFreezeHighOnly-{}".format(env_name, seed)
        TransferLinearFreezeTrainHighOnly(env, val_env, name, epochs, seed)

def train_transfer_fine_tune(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-low_train-{}".format(env_name, seed)
        TransferTrain(env, val_env, name, epochs, True, seed)

def train_transfer_fine_tune_high_only(seed, epochs):
    envs = [Othello_four_by_four, Othello_six_by_six, OthelloScoreEnvironment, OthelloStartingPositionChange]
    validation_envs = [Othello_four_by_fourValidation, Othello_six_by_sixValidation, OthelloScoreEnvironmentValidation, OthelloStartingPositionChangeValidation]
    env_names = ["four_by_four", "six_by_six", "ScoreEnvironment", "StartingPositionChange"]

    for (env_name, env, val_env) in zip(env_names, envs, validation_envs):
        name = "Transfer-{}-only_high-{}".format(env_name, seed)
        TransferTrain(env, val_env, name, epochs, False, seed)

if __name__ == "__main__":
    #"fine_tune", "fine_tune_high_only", "freeze_train_high_only", "freeze_train", "zero_shot", "from_scratch", "from_scratch_high_only"
    tokens = {"fine_tune": train_transfer_fine_tune,
    "fine_tune_high_only": train_transfer_fine_tune_high_only,
    "freeze_train_high_only": train_linear_freeze_shot_high_only,
    "freeze_train": train_linear_freeze_shot,
    "zero_shot": train_zero_shot,
    "from_scratch": train_transfer_from_scratch,
    "from_scratch_high_only": train_transfer_high_scratch}
    
    arguments = sys.argv[1:]
    assert len(arguments) == 3
    assert arguments[0] in tokens
    assert int(arguments[1]) > 0
    assert int(arguments[2]) > 0

    function = tokens[arguments[0]]
    seed = int(arguments[1])
    n_epochs = int(arguments[2])

    function(seed, n_epochs)