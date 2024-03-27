from Models import initialize_networks
from HIRO import train
from Environments import OthelloLearningEnvironment, OthelloEnvironment, OthelloEasyEnvironment
import sys
import tensorflow as tf

def HIROTrain(name, n_epochs, seed=1):
    low_parameters = {
        "batch_N": 256,
        "replay_buffer_size": 25000,
        "update_period": 1,
        "epsilon": 0.05,
        "gamma": 0.0,
        "n_updates": 1,
        "actor_lr": 1e-4,
        "actor_clipvalue": 1.0,
        "start_weights_filepaths": ["HIRO_weights/low_network-1.h5", "HIRO_weights/low_network_target-1.h5"],
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
        "start_weights_filepaths": ["HIRO_weights/high_network-1.h5", "HIRO_weights/high_network_target-1.h5", "HIRO_weights/high_network_critic-1.h5", "HIRO_weights/high_network_critic_target-1.h5"],
        "regularization": 0.0
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "max_random_moves_before_game": 0
    }
    meta_parameters = {
        "n_epochs": n_epochs,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "weights_dir": "drive/MyDrive/othello/HIRO_weights/{}".format(name),
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "latest_player_save_period": 5000,
        "evaluation_period": 25000,
        "opponent_evaluation_games": 0,
        "random_seed": seed,
        "gc_period": 100,
        "ai_low_network_depth": 0,
        "n": 1.0,
        "use_selfplay": False,
        "results_n": 1,
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
        "run_validation": True,
        "is_learning_environment": True,
        "load_low_weights": False,
        "load_high_weights": False,
        "low_freeze_all_but_last": False,
        "high_freeze_all_but_last": False,
        "disable_goal_relabelling": False
    }
    return train(OthelloLearningEnvironment, env_parameters, meta_parameters, low_parameters, high_parameters, initialize_networks)

if __name__ == "__main__":
    arguments = sys.argv[1:]
    
    assert len(arguments) == 2
    assert int(arguments[0]) > 0
    assert int(arguments[1]) > 0 

    seed = int(arguments[0])
    n_epochs = int(arguments[1])

    name = "HIRO_{}_{}".format(seed, n_epochs)

    HIROTrain(name, n_epochs, seed)