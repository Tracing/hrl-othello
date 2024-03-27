#include <iostream>
#include "io.h"

char to_char(colour_t c) {
    if(c == EMPTY) {
        return '.';
    } else if(c == WHITE) {
        return 'O';
    } else {
        return 'X';
    }
}

void print_game(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn) {
    std::cout << (*colour_turn == WHITE ? "White" : "Black") << " turn\n";
    for(int y = N-1; y >= 0; y--) {
        for(int x = 0; x < N; x++) {
            std::cout << " " << to_char((*board)[x][y]);
        }
        std::cout << "\n";
    }
}

void print_game_2(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, colour_t last_move_colour, move_t last_move) {
    ;
}