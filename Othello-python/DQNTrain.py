from Models import get_dqn_network
import Environments as env
import DQN

def dqnTrain(name, seed=1):
    parameters = {
        "n_epochs": 4000,
        "log_dir": "drive/MyDrive/othello/logs/{}".format(name),
        "save_period": 500, 
        "batch_N": 256, 
        "replay_buffer_size": 25000, 
        "gamma": 0.99, 
        "epsilon": 0.05,
        "lr": 1e-4,
        "rho": 1e-4,
        "random_seed": seed,
        "weights_dir": "dqn_weights",
        "evaluation_n": 0,
        "results_n": 200,
        "results_output_dir": "drive/MyDrive/othello/results/{}".format(name),
        "evaluation_output_dir": "drive/MyDrive/othello/evaluation/{}".format(name),
        "evaluation_period": 25,
        "opponent_evaluation_games": 200,
        "l2": 0.00,
        "clipvalue": 1.0,
        "update_period": 1,
        "score_ma": 400,
        "cutoff_ma": 1.1,
        "is_othello_environment": True,
        "gc_period": 100,
        "a_low": -1,
        "a_high": 1,
        "student_required_ma": 0.7,
        "student_required_n": 100,
        "student_past_agents_prob": 0.2,
        "training_level_cutoff": 4,
        "run_validation": False
    }
    env_parameters = {
        "high_network_weights_directory": "HIRO_high_weights",
        "low_network_weights_directory": "HIRO_low_weights",
        "h": 1,
        "previous_opponent_prob": 0.2,
        "max_random_moves_before_game": 0,
        "epsilon": 0.00
    }
    return DQN.dqn(env.OthelloLearningEnvironment, env_parameters, parameters, get_dqn_network)

if __name__ == "__main__":
    for seed in range(1, 11):
        dqnTrain("DQN_{}".format(seed), seed)