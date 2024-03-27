from Agents import Agent, DQN_Agent, HIROAgent, TransferAgent, MCTSAgent, MinimaxAgent
import Environments as envs
import othello_wrapper
import math
import tensorflow as tf
from scipy.stats import ttest_ind, ttest_rel
import numpy as np

def bonferroni_correction(k, alpha):
    return alpha / k

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

def run_environment(environment: envs.OthelloEnvironment, environment_weights, agent: Agent):
    if isinstance(agent, TransferAgent):
        agent.set_goal_weights(environment_weights)

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

    return cum_reward

def produce_statistics(scores1, scores2, alpha):
    p_value = ttest_rel(scores1, scores2, alternative='greater')[1]
    is_significant = p_value <= alpha

    return (p_value, is_significant)

def generate_latex(output_file, n_games, scores, total_scores, p_values):
    #Make statistics for: 
    # overall mean
    # overall stdev
    # each environment mean
    # each environment stdev

    #Make table of:
    # rows: comparison, p-value, significance

    #Make table of:
    # rows: agent_name, total_score, n_games and 95% confidence interval.

    #Per environment, make table of:
    # rows: agent_name, individual mean, n_games and 95% confidence interval.

    z = 1.96

    overall_means = {agent_name: np.mean(total_scores[agent_name]) for agent_name in scores}
    overall_stddevs = {agent_name: np.std(total_scores[agent_name]) / math.sqrt(n_games * 5) for agent_name in scores}
    means = {(agent_name, environment_name): np.mean(scores[agent_name][environment_name]) for agent_name in scores for environment_name in envs}
    stddevs = {(agent_name, environment_name): np.std(scores[agent_name][environment_name]) / math.sqrt(n_games) for agent_name in scores for environment_name in envs}

    latex_lines = ["\\begin{center}", 
                "\\begin{tabular}{| c | c | c |}",
                "\\hline",
                "Agent 1 & Agent 2 & p-value",
                "\\hline",
                "Transfer & DQN & {:.18f}{}".format(p_values["transfer_dqn"][0], "*" if p_values["transfer_dqn"][1] else ""),
                "Transfer & HIRO-A & {:.18f}{}".format(p_values["transfer_HIRO-A"][0], "*" if p_values["transfer_HIRO-A"][1] else ""),
                "HIRO-A & DQN & {:.18f}{}".format(p_values["HIRO-A_dqn"][0], "*" if p_values["HIRO-A_dqn"][1] else ""),
                "\\hline",
                "\\end{tabular}",
                "\\end{center}",

                "\\begin{center}", 
                "\\begin{tabular}{| c | c | c |}",
                "\\hline",
                "Agent Name & Mean Reward & Number of Games",
                "\\hline",
                "Transfer & {:.18f}+-{:.18f} & {}\\\\".format(overall_means["transfer"], overall_stddevs["transfer"] * z, n_games),
                "HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(overall_means["HIRO-A"], overall_stddevs["HIRO-A"] * z, n_games),
                "DQN & {:.18f}+-{:.18f} & {}".format(overall_means["dqn"], overall_stddevs["dqn"] * z, n_games),
                "\\hline",
                "\\end{tabular}",
                "\\end{center}",

                "\\begin{center}", 
                "\\begin{tabular}{| c | c | c | c |}",
                "\\hline",
                "Environment Name & Agent Name & Mean Reward & Number of Games",
                "\\hline",
                "Four-by-Four & Transfer & {:.18f}+-{:.18f} & {}\\\\".format(means[("transfer", "four_by_four")], stddevs[("transfer", "four_by_four")] * z, n_games),
                "Four-by-Four & HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(means[("HIRO-A", "four_by_four")], stddevs[("HIRO-A", "four_by_four")] * z, n_games),
                "Four-by-Four & DQN & {:.18f}+-{:.18f} & {}\\\\".format(means[("dqn", "four_by_four")], stddevs[("dqn", "four_by_four")] * z, n_games),
                "\\hline",
                "Six-by-Six & Transfer & {:.18f}+-{:.18f} & {}\\\\".format(means[("transfer", "six_by_six")], stddevs[("transfer", "six_by_six")] * z, n_games),
                "Six-by-Six & HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(means[("HIRO-A", "six_by_six")], stddevs[("HIRO-A", "six_by_six")] * z, n_games),
                "Six-by-Six & DQN & {:.18f}+-{:.18f} & {}\\\\".format(means[("dqn", "six_by_six")], stddevs[("dqn", "six_by_six")] * z, n_games),
                "\\hline",
                "Inverse & Transfer & {:.18f}+-{:.18f} & {}\\\\".format(means[("transfer", "inverse")], stddevs[("transfer", "inverse")] * z, n_games),
                "Inverse & HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(means[("HIRO-A", "inverse")], stddevs[("HIRO-A", "inverse")] * z, n_games),
                "Inverse & DQN & {:.18f}+-{:.18f} & {}\\\\".format(means[("dqn", "inverse")], stddevs[("dqn", "inverse")] * z, n_games),
                "\\hline",
                "Score & Transfer & {:.18f}+-{:.18f} & {}\\\\".format(means[("transfer", "score")], stddevs[("transfer", "score")] * z, n_games),
                "Score & HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(means[("HIRO-A", "score")], stddevs[("HIRO-A", "score")] * z, n_games),
                "Score & DQN & {:.18f}+-{:.18f} & {}\\\\".format(means[("dqn", "score")], stddevs[("dqn", "score")] * z, n_games),
                "\\hline",
                "Starting Position Change & Transfer & {:.18f}+-{:.18f} & {}\\\\".format(means[("transfer", "starting_position_change")], stddevs[("transfer", "starting_position_change")] * z, n_games),
                "Starting Position Change & HIRO-A & {:.18f}+-{:.18f} & {}\\\\".format(means[("HIRO-A", "starting_position_change")], stddevs[("HIRO-A", "starting_position_change")] * z, n_games),
                "Starting Position Change & DQN & {:.18f}+-{:.18f} & {}".format(means[("dqn", "starting_position_change")], stddevs[("dqn", "starting_position_change")] * z, n_games),
                "\\hline",
                "\\end{tabular}",
                "\\end{center}",
                ]

    string = "\n".join(latex_lines)

    with open(output_file, "w") as f:
        f.write(string)

def run_n_environment(n_games: int, environment: envs.OthelloEnvironment, environment_weights, agent: Agent):
    score = []
    for _ in range(n_games):
        score.append(run_environment(environment, environment_weights, agent))
    return score

if __name__ == "__main__":
    n_games = 1000
    alpha = 0.05
    adjusted_alpha = bonferroni_correction(3, alpha)
    seed = 1
    output_file = "./transferability.txt"

    dqn_weights = "./transferability_weights/dqn.h5"
    low_weights = "./transferability_weights/low_network.h5"
    high_weights = "./transferability_weights/high_network.h5"

    dqn_agent = DQN_Agent(0.05, dqn_weights)
    HIRO_A_agent = HIROAgent(high_weights, low_weights, 0.0, 0.0)
    transfer_agent = TransferAgent(high_weights, low_weights, 0.0, 0.0)

    parameters = get_parameters()

    agents = {
        "dqn": dqn_agent,
        "HIRO-A": HIRO_A_agent,
        "transfer": transfer_agent
    }

    scores = {
        "dqn": {"four_by_four": [], "six_by_six": [], "inverse": [], "score": [], "starting_position_change": []},
        "HIRO-A": {"four_by_four": [], "six_by_six": [], "inverse": [], "score": [], "starting_position_change": []},
        "transfer": {"four_by_four": [], "six_by_six": [], "inverse": [], "score": [], "starting_position_change": []}
    }

    total_scores = {
        "dqn": [],
        "HIRO-A": [],
        "transfer": []
    }

    envs = {
        "four_by_four": envs.Othello_four_by_four(parameters),
        "six_by_six": envs.Othello_six_by_six(parameters),
        "inverse": envs.OthelloInverseEnvironment(parameters),
        "score": envs.OthelloScoreEnvironment(parameters),
        "starting_position_change": envs.OthelloStartingPositionChange(parameters)
    }

    for (agent_name, agent) in agents.items():
        for (environment_name, environment) in envs.items():
            print("Running {} games with agent {} in environment {}".format(n_games, agent_name, environment))
            scores[agent_name][environment_name] = run_n_environment(n_games, environment, env_weights[environment_name], agent)

    for agent_name in total_scores:
        score = []
        for env_name in envs:
            score.extend(scores[agent_name][env_name])
        total_scores[agent_name] = score
    
    (HIRO_A_dqn_p_value, HIRO_A_dqn_significant) = produce_statistics(total_scores["HIRO-A"], total_scores["dqn"], adjusted_alpha)
    (transfer_dqn_p_value, transfer_dqn_significant) = produce_statistics(total_scores["transfer"], total_scores["dqn"], adjusted_alpha)
    (transfer_HIRO_A_p_value, transfer_HIRO_A_significant) = produce_statistics(total_scores["transfer"], total_scores["HIRO-A"], adjusted_alpha)

    p_values = {"HIRO-A_dqn": (HIRO_A_dqn_p_value, HIRO_A_dqn_significant),
                "transfer_dqn": (transfer_dqn_p_value, transfer_dqn_significant),
                "transfer_HIRO-A": (transfer_HIRO_A_p_value, transfer_HIRO_A_significant)}

    print("adjusted alpha = {:.4f}".format(adjusted_alpha))
    print("HIRO-A_dqn_p_value = {:.5f}".format(HIRO_A_dqn_p_value))
    print("transfer_dqn_p_value = {:.5f}".format(transfer_dqn_p_value))
    print("transfer_HIRO-A_p_value = {:.5f}".format(transfer_HIRO_A_p_value))

    generate_latex(output_file, n_games, scores, total_scores, p_values)

    #Make statistics for: 
    # overall mean
    # overall stdev
    # each environment mean
    # each environment stdev

    #Make table of:
    # rows: comparison, p-value, significance

    #Make table of:
    # rows: agent_name, total_score, n_games and 95% confidence interval.

    #Make table of:
    # rows: agent_name, individual mean, n_games and 95% confidence interval.