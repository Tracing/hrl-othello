#include "constants.h"
#include <vector>

#ifndef ENGINE_H
#define ENGINE_H

void flip_counters(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, char delta_x, char delta_y);
bool is_flipable(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, char delta_x, char delta_y);
void get_colour_moves(std::vector<std::vector<colour_t>>* board, colour_t c, std::vector<move_t>* moves);
void get_moves_for_square(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, std::vector<move_t>* moves);
void make_single_move(std::vector<std::vector<colour_t>>* board, move_t m, colour_t c);

std::vector<std::vector<colour_t>> new_board();
void get_all_moves(std::vector<std::vector<colour_t>>* board, std::vector<move_t>* whiteMoves, std::vector<move_t>* blackMoves);
colour_t determine_winner(std::vector<std::vector<colour_t>>* board);
std::vector<std::vector<colour_t>> copy_board(std::vector<std::vector<colour_t>>* board);
Move get_null_move();
Move get_move(char x, char y);
bool game_has_ended(std::vector<std::vector<colour_t>>* board, colour_t colour_turn);
void get_moves(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, std::vector<move_t>* whiteMoves, std::vector<move_t>* blackMoves);
void make_move(std::vector<std::vector<colour_t>>* board, move_t whiteMove, move_t blackMove, colour_t* colour_turn, colour_t* winner);
std::vector<std::vector<colour_t>> copy_board(std::vector<std::vector<colour_t>>* board);
#endif