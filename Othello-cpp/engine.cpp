#include "constants.h"
#include "engine.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

std::vector<std::vector<colour_t>> new_board() {
    std::vector<std::vector<colour_t>> board;
    int x = (int) round((float) N / 2.0);
    for(int i = 0; i < N; i++) {
        board.push_back(std::vector<colour_t>());
        for(int j = 0; j < N; j++) {
            board[i].push_back(EMPTY);
        }
    }
    board[x-1][x-1] = BLACK;
    board[x-1][x] = WHITE;
    board[x][x-1] = WHITE;
    board[x][x] = BLACK;
    return board;
}

std::vector<std::vector<colour_t>> copy_board(std::vector<std::vector<colour_t>>* board) {
    std::vector<std::vector<colour_t>> new_board;
    for(int i = 0; i < N; i++) {
        new_board.push_back(std::vector<colour_t>());
        for(int j = 0; j < N; j++) {
            new_board[i].push_back((*board)[i][j]);
        }
    }
    return new_board;
}

void flip_counters(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, char delta_x, char delta_y) {
    bool stop = !is_flipable(board, c, x, y, delta_x, delta_y);
    while (!stop) {
        x = x + delta_x;
        y = y + delta_y;
        if(x < N && y < N && x >= 0 && y >= 0) {
            if((*board)[x][y] == c || (*board)[x][y] == EMPTY || (*board)[x][y] == BLOCKING) {
                stop = true;
            } else {
                (*board)[x][y] = c;
            }
        } else {
            stop = true;
        }
    }
}

bool is_flipable(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, char delta_x, char delta_y) {
    bool stop = false;
    bool end_point = false;
    bool intermediate = false;
    while (!stop) {
        x = x + delta_x;
        y = y + delta_y;
        if(x < N && y < N && x >= 0 && y >= 0) {
            if((*board)[x][y] == EMPTY) {
                end_point = false;
                stop = true;
            } else if((*board)[x][y] == BLOCKING) {
                end_point = false;
                stop = true;
            } else if((*board)[x][y] == c) {
                end_point = true;
                stop = true;
            } else {
                intermediate = true;
            }
        } else {
            stop = true;
        }
    }
    return end_point && intermediate;
}

void make_single_move(std::vector<std::vector<colour_t>>* board, move_t m, colour_t c) {
    (*board)[m.x][m.y] = c;
    //Flip counters
    flip_counters(board, c, m.x, m.y, 1, 0);
    flip_counters(board, c, m.x, m.y, 0, 1);
    flip_counters(board, c, m.x, m.y, 1, 1);
    flip_counters(board, c, m.x, m.y, -1, 0);
    flip_counters(board, c, m.x, m.y, 0, -1);
    flip_counters(board, c, m.x, m.y, -1, -1);
    flip_counters(board, c, m.x, m.y, -1, 1);
    flip_counters(board, c, m.x, m.y, 1, -1);
}

bool game_has_ended(std::vector<std::vector<colour_t>>* board, colour_t player_turn) {
    //Check if board is full
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    char x = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] != EMPTY) {
                x++;
            }
        }
    }
    if(x == N * N) {
        return true;
    } else {
        //Check if neither player can not move
        get_all_moves(board, &whiteMoves, &blackMoves);
        return whiteMoves.at(0).x == NULL_VALUE && blackMoves.at(0).x == NULL_VALUE;
    }   
}

colour_t determine_winner(std::vector<std::vector<colour_t>>* board) {
    char white_pieces = 0;
    char black_pieces = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] == WHITE) {
                white_pieces++;
            } else if((*board)[i][j] == BLACK) {
                black_pieces++;
            }
        }
    }
    if(white_pieces > black_pieces) {
        return WHITE;
    } else if(white_pieces == black_pieces) {
        return DRAW;
    } else {
        return BLACK;
    }
}

Move get_null_move() {
    Move null_move;
    null_move.x = NULL_VALUE;
    null_move.y = NULL_VALUE;
    return null_move;
}

Move get_move(char x, char y) {
    Move move;
    move.x = x;
    move.y = y;
    return move;
}

void get_all_moves(std::vector<std::vector<colour_t>>* board, std::vector<move_t>* whiteMoves, std::vector<move_t>* blackMoves) {
    whiteMoves->clear();
    blackMoves->clear();
    get_colour_moves(board, WHITE, whiteMoves);
    get_colour_moves(board, BLACK, blackMoves);
}

void get_moves(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, std::vector<move_t>* whiteMoves, std::vector<move_t>* blackMoves) {
    whiteMoves->clear();
    blackMoves->clear();
    if(*colour_turn == WHITE) {
        blackMoves->push_back(get_null_move());
        get_colour_moves(board, *colour_turn, whiteMoves);
    } else {
        whiteMoves->push_back(get_null_move());
        get_colour_moves(board, *colour_turn, blackMoves);
    }
}

void get_colour_moves(std::vector<std::vector<colour_t>>* board, colour_t c, std::vector<move_t>* moves) {
    //Go through every square on the game board and see whether a tile can be placed there
    for(int x = 0; x < N; x++) {
        for(int y = 0; y < N; y++) {
            get_moves_for_square(board, c, y, x, moves);
        }
    }
    if(moves->size() == 0) {
        moves->push_back(get_null_move());
    }
}

void get_moves_for_square(std::vector<std::vector<colour_t>>* board, colour_t c, char x, char y, std::vector<move_t>* moves) {
    //Add all moves that involve placing a counter on square at row y and column x for colour_t c
    move_t move;
    bool is_counter_placable = (is_flipable(board, c, x, y, 1, 0) ||
    is_flipable(board, c, x, y, 0, 1) ||
    is_flipable(board, c, x, y, 1, 1) ||
    is_flipable(board, c, x, y, -1, 0) ||
    is_flipable(board, c, x, y, 0, -1) ||
    is_flipable(board, c, x, y, -1, -1) ||
    is_flipable(board, c, x, y, -1, 1) ||
    is_flipable(board, c, x, y, 1, -1)) && (*board)[x][y] == EMPTY;
    
    move.x = x;
    move.y = y;
    if(is_counter_placable) {
        moves->push_back(move);
    }
}

void make_move(std::vector<std::vector<colour_t>>* board, move_t whiteMove, move_t blackMove, colour_t* colour_turn, colour_t* winner) {
    assert(whiteMove.x < N);
    assert(whiteMove.y < N);
    assert(blackMove.x < N);
    assert(blackMove.y < N);
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;

    if(whiteMove.x != NULL_VALUE && whiteMove.y != NULL_VALUE) {
        make_single_move(board, whiteMove, *colour_turn);
    }
    if(blackMove.x != NULL_VALUE && blackMove.y > NULL_VALUE) {
        make_single_move(board, blackMove, *colour_turn);
    }
    
    //Alter game state
    *colour_turn = (*colour_turn == WHITE) ? BLACK : WHITE;
    if(game_has_ended(board, *colour_turn)) {
        *winner = determine_winner(board);
    } else {
        get_all_moves(board, &whiteMoves, &blackMoves);
        if(*colour_turn == WHITE && whiteMoves.at(0).x == NULL_VALUE) {
            *colour_turn = BLACK;
        } else if(*colour_turn == BLACK && blackMoves.at(0).x == NULL_VALUE) {
            *colour_turn = WHITE;
        }
    }
}