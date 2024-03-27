#include "constants.h"
#include "engine.h"
#include "io.h"
#include <vector>

#ifndef OBJECTS_OTHELLO_H
#define OBJECTS_OTHELLO_H

class OthelloGame {
    public:
        ~OthelloGame();
        OthelloGame();
        std::vector<std::vector<colour_t>> board;
        colour_t colour_turn;
        colour_t winner;
        std::vector<move_t> whiteMoves;
        std::vector<move_t> blackMoves;
        std::vector<std::vector<colour_t>>* get_board();
        void set_square(int x, int y, colour_t colour);
        colour_t at_square(int x, int y);
        bool game_has_ended();
        void reset();
        void do_move(move_t whiteMove, move_t blackMove);
        OthelloGame copy();
};

#endif