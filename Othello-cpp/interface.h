#include "ai.h"
#include <ncurses.h>

#ifndef INTERFACE_H
#define INTERFACE_H
void run_random_game();
void run_minimax_game(int d);
void run_game(ai_t ai_white, ai_t ai_black, int minimax_depth_white, int minimax_depth_black, int mcts_n_white, int mcts_n_black, float mcts_C_white, float mcts_C_black);

#endif