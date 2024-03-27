#include "constants.h"
#include "engine.h"

#ifndef AI_H
#define AI_H

#define AI_RANDOM 0
#define AI_MINIMAX 1
#define AI_MCTS 2

typedef int ai_t;

typedef struct mcts_tree {
    std::vector<std::vector<colour_t>> board;
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<mcts_tree> children;
    colour_t colour_turn;
    colour_t winner;
    float visits;
    float score;
    mcts_tree* parent;
    bool terminal;
} mcts_tree_t;

move_t get_minimax_move(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, bool use_goal, colour_t* colour_turn, int depth, unsigned int seed);
move_t get_mcts_move(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, int n, float C, unsigned int seed);
move_t get_monte_carlo_move(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, colour_t for_colour, int n);
float heuristic_fn(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, bool use_goal, colour_t colour_turn, colour_t for_colour);
colour_t play_random_game(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn);

#endif