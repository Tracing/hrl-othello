import math
import othello
from othello_wrapper import OthelloGame

game = OthelloGame()
moves = game.get_moves_2()

n_samples = 1000
mcts_moves = []
minimax_moves = []
minimax_depth = 15

for i in range(n_samples):
    move = othello.get_mcts_move(game.game.board, game.get_player_turn(), 10000, math.sqrt(2))
    mcts_moves.append(move)
    move = othello.get_minimax_move(game.game.board, game.get_player_turn(), minimax_depth)
    minimax_moves.append(move)

with open("statistics.csv") as f:
    f.write()

