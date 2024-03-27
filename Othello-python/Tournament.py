from Agents import Agent, MinimaxAgent, MCTSAgent, DQN_Agent, HIROAgent
from misc_functions import wipe_dir
from othello_wrapper import OthelloGame
from constants import WHITE, BLACK, NULL
import numpy as np
import math
import os
import random
import tensorflow as tf

def play_game(game, agent_white: Agent, agent_black: Agent):
    game.reset()
    
    agent_white.new_game(game, WHITE)
    agent_black.new_game(game, BLACK)

    while not game.game_has_ended():
        (whiteMoves, blackMoves) = game.get_moves()
        if game.get_player_turn() == WHITE:
            move = agent_white.get_move(game)
            agent_white.record_pre_move(game)
            game.make_move_2(move)
        else:
            move = agent_black.get_move(game)
            agent_black.record_pre_move(game)
            game.make_move_2(move)

    winner = game.get_winner()
    if winner == WHITE:
        scores = [1, 0]
    elif winner == BLACK:
        scores = [0, 1]
    else:
        scores = [0.5, 0.5]

    return scores

def play_game_2(game, agent1: Agent, agent2: Agent):
    agent1_white = random.random() < 0.5
    scores = [0, 0]
    if agent1_white:
        scores = play_game(game, agent1, agent2)
    else:
        _scores = play_game(game, agent2, agent1)
        (scores[0], scores[1]) = (_scores[1], _scores[0])
    
    return scores

def play_n_games(n: int, agent1: Agent, agent2: Agent, log_directory1, log_directory2, verbose=False, include_stddev=False):
    if verbose:
        for directory in [log_directory1, log_directory2]:
            wipe_dir(directory)

    game = OthelloGame()
    scores = [0, 0]
    scores_statistics = []

    if verbose:
        agent1.set_log_directory(log_directory1)
        agent2.set_log_directory(log_directory2)
    else:
        agent1.set_logging_off()
        agent2.set_logging_off()

    for i in range(n):
        _scores = play_game_2(game, agent1, agent2)
        scores[0] += _scores[0]
        scores_statistics.append(_scores[0])
        scores[1] += _scores[1]
        if verbose:
            print("Game {}/{}".format(i+1, n))
    if verbose:
        print("Agent 1 score: {}".format(scores[0]))
        print("Agent 2 score: {}".format(scores[1]))

    if not include_stddev:
        return scores[0]
    else:
        return (scores[0], np.std(scores_statistics))

if __name__ == "__main__":
    n = 1000
    outfile_name = "tournament_results.csv"

    hiroAgent = HIROAgent("tournament_weights/high_network.h5", "tournament_weights/low_network.h5", 0.05, 0)
    agent = Agent()
    minimaxAgent_1 = MinimaxAgent(1)
    minimaxAgent_2 = MinimaxAgent(2)
    mctsAgent = MCTSAgent(50, math.sqrt(2))
    dqnAgent = DQN_Agent(0.05, "tournament_weights/dqn.h5")

    games = ["hiro_agent", "hiro_minimax_1", "hiro_minimax_2", "hiro_mcts", "hiro_dqn_agent", "agent_minimax_1", "agent_minimax_2", "agent_mcts", "agent_dqn_agent", "minimax_1_minimax_2", "minimax_1_mcts", "minimax_1_dqn_agent", "minimax_2_mcts", "minimax_2_dqn_agent", "mcts_dqn_agent"]
    wins = []

    wins.append(play_n_games(n, hiroAgent, agent, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, hiroAgent, minimaxAgent_1, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, hiroAgent, minimaxAgent_2, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, hiroAgent, mctsAgent, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, hiroAgent, dqnAgent, "HIRO_log", "Random_log", verbose=False))

    wins.append(play_n_games(n, agent, minimaxAgent_1, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, agent, minimaxAgent_2, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, agent, mctsAgent, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, agent, dqnAgent, "HIRO_log", "Random_log", verbose=False))

    wins.append(play_n_games(n, minimaxAgent_1, minimaxAgent_2, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, minimaxAgent_1, mctsAgent, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, minimaxAgent_1, dqnAgent, "HIRO_log", "Random_log", verbose=False))

    wins.append(play_n_games(n, minimaxAgent_2, mctsAgent, "HIRO_log", "Random_log", verbose=False))
    wins.append(play_n_games(n, minimaxAgent_2, dqnAgent, "HIRO_log", "Random_log", verbose=False))

    wins.append(play_n_games(n, mctsAgent, dqnAgent, "HIRO_log", "Random_log", verbose=False))

    lines = ["matchup, score"]
    for (name, wins) in zip(games, wins):
        print("{} matchup had a score of {:.1f}".format(name, wins))
        lines.append("{}, {:.1f}".format(name, wins))

    with open(outfile_name, "w") as f:
        f.write("\n".join(lines))

    #play_n_games(500, hiroAgent, minimaxAgent)