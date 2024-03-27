from Agents import Agent, HIROAgent
import Environments as envs
import numpy as np
import random

def get_parameters():
    parameters = {
        "high_network_weights_directory": "HIRO-A_high_weights",
        "low_network_weights_directory": "HIRO-A_low_weights",
        "h": 1,
        "previous_opponent_prob": 0.2,
        "max_random_moves_before_game": 0,
        "epsilon": 0.00
    }
    return parameters

def run_environment(environment: envs.OthelloEnvironment, agent: Agent):
    environment.reset()
    s = environment.state
    terminal = False
    cum_reward = 0
    while not terminal:
        #Select action
        a = agent.get_action(environment.game)
        #Execute action
        (_, r, d) = environment.step(int(a))
        cum_reward += r[0, 0]
        terminal = d > 0.999

    return float(cum_reward)

def run_n_environment(n_games: int, environment: envs.OthelloEnvironment, agent: Agent, outfile_path):
    scores = []
    for _ in range(n_games):
        scores.append(str(run_environment(environment, agent)))
    
    with open(outfile_path, "w") as f:
        f.write(",".join(scores))

if __name__ == "__main__":
    n_games = 1000
    low_weights = "drive/MyDrive/othello/HIRO_weights/low_network-9.h5"
    high_weights = "drive/MyDrive/othello/HIRO_weights/high_network-9.h5"

    four_by_four_low_weights = "drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-low_train-1/low_network-9.h5"
    four_by_four_high_weights = "drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-low_train-1/high_network-9.h5"
    six_by_six_low_weights = "drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-low_train-1/low_network-9.h5"
    six_by_six_high_weights = "drive/MyDrive/othello/HIRO_weights/high_network-9.h5"
    score_low_weights = "drive/MyDrive/othello/HIRO_weights/low_network-9.h5"
    score_high_weights = "drive/MyDrive/othello/HIRO_weights/high_network-9.h5"
    starting_position_change_low_weights = "drive/MyDrive/othello/HIRO_weights/low_network-9.h5"
    starting_position_change_high_weights = "drive/MyDrive/othello/HIRO_weights/high_network-9.h5"

    transerability_results_path = "drive/MyDrive/othello/transerability_results"

    random.seed(1)

    parameters = get_parameters()

    envs = {
        "four_by_four": envs.Othello_four_by_four(parameters),
        "six_by_six": envs.Othello_six_by_six(parameters),
        "score": envs.OthelloScoreEnvironment(parameters),
        "starting_position_change": envs.OthelloStartingPositionChange(parameters)
    }
    agents = {
        "vanilla": HIROAgent("drive/MyDrive/othello/HIRO_weights/high_network-9.h5", "drive/MyDrive/othello/HIRO_weights/low_network-9.h5", 0.0, 0.0),
        "four_by_four_low": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-low_train-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-low_train-1/low_network-2.h5", 0.0, 0.0),
        "four_by_four_only_high": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-only_high-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-four_by_four-only_high-1/low_network-2.h5", 0.0, 0.0),

        "six_by_six_low": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-six_by_six-low_train-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-six_by_six-low_train-1/low_network-2.h5", 0.0, 0.0),
        "six_by_six_only_high": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-six_by_six-only_high-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-six_by_six-only_high-1/low_network-2.h5", 0.0, 0.0),

        "score_low": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-ScoreEnvironment-low_train-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-ScoreEnvironment-low_train-1/low_network-2.h5", 0.0, 0.0),
        "score_only_high": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-ScoreEnvironment-only_high-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-ScoreEnvironment-only_high-1/low_network-2.h5", 0.0, 0.0),

        "starting_position_change_low": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-StartingPositionChange-low_train-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-StartingPositionChange-low_train-1/low_network-2.h5", 0.0, 0.0),
        "starting_position_change_only_high": HIROAgent("drive/MyDrive/othello/HIRO_weights_Transfer-StartingPositionChange-only_high-1/high_network-2.h5", "drive/MyDrive/othello/HIRO_weights_Transfer-StartingPositionChange-only_high-1/low_network-2.h5", 0.0, 0.0),
        }

    matches = [("vanilla", "four_by_four"), ("vanilla", "six_by_six"), ("vanilla", "score"), ("vanilla", "starting_position_change"), ]
    
    for (environment_name, environment) in envs.items():
        for (agent_name, agent) in agents.items():
            print("Running {} games with agent {} in environment {}".format(n_games, agent_name, environment_name))
            run_n_environment(n_games, environment, agent, "{}/{}-{}.csv".format(transerability_results_path, environment_name, agent_name))
        