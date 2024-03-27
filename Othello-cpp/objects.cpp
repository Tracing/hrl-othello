#include "constants.h"
#include "objects.h"

void OthelloGame::do_move(move_t whiteMove, move_t blackMove) {
    make_move(&board, whiteMove, blackMove, &colour_turn, &winner);
    if(winner == NONE) {
        get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    } else {
        whiteMoves.clear();
        blackMoves.clear();
    }
    if(colour_turn == WHITE) {
        if(whiteMoves[0].x == NULL_VALUE) {
            do_move(get_null_move(), get_null_move());
        }
    } else if(colour_turn == BLACK) {
        if(blackMoves[0].x == NULL_VALUE) {
            do_move(get_null_move(), get_null_move());
        }
    }
    
}

OthelloGame::OthelloGame() {
    reset();
}

OthelloGame::~OthelloGame() {
    ;
}

std::vector<std::vector<colour_t>>* OthelloGame::get_board() {
    return &board;
}

colour_t OthelloGame::at_square(int x, int y) {
    return board[x][y];
}

void OthelloGame::set_square(int x, int y, colour_t colour) {
    board[x][y] = colour;
}

void OthelloGame::reset() {
    board = new_board();
    colour_turn = BLACK;
    winner = NONE;
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
}

OthelloGame OthelloGame::copy() {
    OthelloGame game;

    game.blackMoves.clear();
    for(int i = 0; i < blackMoves.size(); i++) {
        game.blackMoves.push_back(blackMoves[i]);
    }

    game.whiteMoves.clear();
    for(int i = 0; i < whiteMoves.size(); i++) {
        game.whiteMoves.push_back(whiteMoves[i]);
    }
    
    game.board = copy_board(&board);
    game.colour_turn = colour_turn;
    game.winner = winner;

    return game;
}

bool OthelloGame::game_has_ended() {
    return winner != NONE;
}