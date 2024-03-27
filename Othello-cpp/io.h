#include "constants.h"
#include <vector>

#ifndef IO_H
#define IO_H

void print_game(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn);
void print_game_2(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, colour_t last_move_colour, move_t last_move);

#endif