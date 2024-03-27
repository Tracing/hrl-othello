#include "ai.h"
#include "constants.h"
#include "engine.h"
#include "interface.h"
#include <curses.h>
#include <string>
#include <cstdlib>
#include <vector>

std::string to_str(colour_t c) {
    if(c == EMPTY) {
        return ".";
    } else if(c == WHITE) {
        return "o";
    } else {
        return "x";
    }
}

std::string to_u_str(colour_t c) {
    if(c == EMPTY) {
        return ".";
    } else if(c == WHITE) {
        return "O";
    } else {
        return "X";
    }
}

void output_game(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, move_t blackMove, move_t whiteMove) {
    clear();
    addstr(*colour_turn == WHITE ? "White" : "Black"); 
    addstr(" turn\n");
    for(int y = N-1; y >= 0; y--) {
        for(int x = 0; x < N; x++) {
            addstr(" ");
            if((blackMove.x == x && blackMove.y == y) || (whiteMove.x == x && whiteMove.y == y)) {
                addstr(to_u_str((*board)[x][y]).c_str());
            } else {
                addstr(to_str((*board)[x][y]).c_str());
            }
        }
        addstr("\n");
    }
}

void output_winner(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, colour_t* winner, move_t blackMove, move_t whiteMove) {
    output_game(board, colour_turn, blackMove, whiteMove);
    addstr(*winner == DRAW ? "DRAW!" : (*winner == WHITE ? "WHITE wins!": "BLACK wins!"));
    addstr("\n");
}

void run_random_game() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    std::vector<std::vector<colour_t>> board = new_board();
    move_t whiteMove = get_null_move();
    move_t blackMove = get_null_move();
    int i, j;

    initscr();
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);

    while(!game_has_ended(&board, colour_turn)) {
        clear();
        output_game(&board, &colour_turn, blackMove, whiteMove);
        refresh();
        getch();

        get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
        i = rand() % whiteMoves.size();
        j = rand() % blackMoves.size();
        whiteMove = whiteMoves[i];
        blackMove = blackMoves[j];
        make_move(&board, whiteMove, blackMove, &colour_turn, &winner);
    }
    clear();
    output_winner(&board, &colour_turn, &winner, blackMove, whiteMove);
    refresh();
    getch();

    endwin();
}

void run_minimax_game(int d) {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    std::vector<std::vector<colour_t>> board = new_board();
    std::vector<float> goal;
    move_t whiteMove = get_null_move();
    move_t blackMove = get_null_move();
    int i, j;

    initscr();
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);

    while(!game_has_ended(&board, colour_turn)) {
        clear();
        output_game(&board, &colour_turn, blackMove, whiteMove);
        refresh();
        getch();

        get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
        if(colour_turn == WHITE) {
            whiteMove = get_minimax_move(&board, &goal, false, &colour_turn, d, 1);
            blackMove = blackMoves[0];
        } else {
            whiteMove = whiteMoves[0];
            blackMove = get_minimax_move(&board, &goal, false, &colour_turn, d, 1);
        }
        make_move(&board, whiteMove, blackMove, &colour_turn, &winner);
    }
    clear();
    output_winner(&board, &colour_turn, &winner, blackMove, whiteMove);
    refresh();
    getch();

    endwin();
}

void run_game(ai_t ai_white, ai_t ai_black, int minimax_depth_white, int minimax_depth_black, int mcts_n_white, int mcts_n_black, float mcts_C_white, float mcts_C_black) {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    std::vector<std::vector<colour_t>> board = new_board();
    std::vector<float> goal;
    move_t whiteMove = get_null_move();
    move_t blackMove = get_null_move();
    int i, j;

    initscr();
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);

    while(!game_has_ended(&board, colour_turn)) {
        clear();
        output_game(&board, &colour_turn, blackMove, whiteMove);
        refresh();
        getch();

        get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
        if(colour_turn == WHITE) {
            if(ai_white == AI_RANDOM) {
                whiteMove = whiteMoves[rand() % whiteMoves.size()];
            } else if(ai_white == AI_MINIMAX) {
                whiteMove = get_minimax_move(&board, &goal, false, &colour_turn, minimax_depth_white, 1);
            } else {
                whiteMove = get_mcts_move(&board, &colour_turn, mcts_n_white, mcts_C_white, 1);
            }
            blackMove = blackMoves[0];
        } else {
            if(ai_black == AI_RANDOM) {
                blackMove = blackMoves[rand() % blackMoves.size()];
            } else if(ai_black == AI_MINIMAX) {
                blackMove = get_minimax_move(&board, &goal, false, &colour_turn, minimax_depth_black, 1);
            } else {
                blackMove = get_mcts_move(&board, &colour_turn, mcts_n_black, mcts_C_black, 1);                
            }
            whiteMove = whiteMoves[0];
        }
        make_move(&board, whiteMove, blackMove, &colour_turn, &winner);
    }
    clear();
    output_winner(&board, &colour_turn, &winner, blackMove, whiteMove);
    refresh();
    getch();

    endwin();
}