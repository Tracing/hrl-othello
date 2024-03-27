from Agents import Agent, MinimaxAgent, MCTSAgent, HIROAgent, DQN_Agent
import misc_functions
import math
import Tournament

def create_results_HIRO(meta_parameters, n, high_network_weights, low_network_weights, low_epsilon, high_epsilon, output_dir="./evaluation_logs"):
    hiroAgent = HIROAgent(high_network_weights, low_network_weights, low_epsilon, high_epsilon, meta_parameters["ai_low_network_depth"], meta_parameters["random_seed"])
    agent = Agent()
    minimax_1 = MinimaxAgent(1)
    minimax_2 = MinimaxAgent(2)
    minimax_3 = MinimaxAgent(3)
    mcts_100 = MCTSAgent(100, math.sqrt(2))

    agents = {
        "hiroAgent": hiroAgent,
        "agent": agent,
        #"minimax_1": minimax_1,
        #"minimax_2": minimax_2,
        #"minimax_3": minimax_3,
        #"mcts_100": mcts_100
    }

    results = {
        ("hiroAgent", "agent"): 0,
        #("hiroAgent", "minimax_1"): 0,
        #("hiroAgent", "minimax_2"): 0,
        #("hiroAgent", "minimax_3"): 0,
        #("hiroAgent", "mcts_100"): 0,
    }

    return create_results(n, agents, results, output_dir)

def create_results_DQN(n, weights, output_dir="./evaluation_logs"):
    dqnAgent = DQN_Agent(0.05, weights)
    agent = Agent()
    minimax_1 = MinimaxAgent(1)
    minimax_2 = MinimaxAgent(2)
    minimax_3 = MinimaxAgent(3)
    mcts_100 = MCTSAgent(100, math.sqrt(2))

    agents = {
        "dqnAgent": dqnAgent,
        "agent": agent,
        #"minimax_1": minimax_1,
        #"minimax_2": minimax_2,
        #"minimax_3": minimax_3,
        #"mcts_100": mcts_100
    }

    results = {
        ("dqnAgent", "agent"): 0,
        #("ddpgAgent", "minimax_1"): 0,
        #("ddpgAgent", "minimax_2"): 0,
        #("ddpgAgent", "minimax_3"): 0,
        #("ddpgAgent", "mcts_100"): 0,
    }

    return create_results(n, agents, results, output_dir)

def score_vs_random_dqn(n_games, network):
    dqn_agent = DQN_Agent(0.05, network=network)
    agent = Agent()
    result = Tournament.play_n_games(n_games, dqn_agent, agent, "./tmp/1", "./tmp/2")
    score = 1 if result > 0.999 else -1 if result < 0.01 else 0
    return score


def get_best_agent_dqn(weights, network_N, evaluation_n, evaluation_agents):
    best_score = float('-inf')
    best_weights = None
    if evaluation_n == 0:
        best_weights = "{}/dqn-{}.h5".format(weights, network_N)
    else:
        for i in range(1, network_N+1):
            print("Evaluating agent {}/{}".format(i, network_N))
            weights_path = "{}/dqn-{}.h5".format(weights, i)
            dqn_agent = DQN_Agent(0.05, weights_path)
            score = 0
            for agent in evaluation_agents:
                score += Tournament.play_n_games(evaluation_n, dqn_agent, agent, "./tmp/1", "./tmp/2")
            if score > best_score:
                best_weights = weights_path
                best_score = score
    print("Chosen set of DQN weights were {}".format(best_weights))
    return best_weights

def create_results(n, agents, results, output_dir="./evaluation_logs/HIRO"):
    keys = list(results.keys())
    new_results = {}
    for (agent1_name, agent2_name) in keys:
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        log_directory1 = "{}/{}/{}".format(output_dir, agent1_name, agent2_name)
        log_directory2 = "{}/tmp".format(output_dir)
        new_results[(agent1_name, agent2_name)] = Tournament.play_n_games(n, agent1, agent2, log_directory1, log_directory2)
    return new_results

def write_results(output_file_path, results, n):
    lines = ["agent1_name, agent2_name, score, n\n"]
    for (agent1_name, agent2_name) in results:
        score = results[(agent1_name, agent2_name)]
        lines.append("{}, {}, {}, {}\n".format(agent1_name, agent2_name, score, n))
    
    with open(output_file_path, "w") as f:
        f.writelines(lines)