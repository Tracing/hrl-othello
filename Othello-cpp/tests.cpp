#include "constants.h"
#include "engine.h"
#include "io.h"
#include "tests.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

/*bool move_in_move_array(move_t move, std::vector<move_t>* moves) {
    //Checks if move in move list
    std::vector<move_t>::iterator it;
    for(it = moves->begin(); it != moves->end(); it++) {
        if(move.x == it->x && move.y == it->y) {
            return true;
        }
    }
    return false;
}

bool move_check(std::vector<move_t>* moves1, std::vector<move_t>* moves2) {
    //Checks if move lists are the same
    std::vector<move_t>::iterator it;
    move_t move;
    if(moves1->size() != moves2->size()) {
        return false;
    } else {
        for(it = moves1->begin(); it != moves1->end(); it++) {
            move.x = it->x;
            move.y = it->y;
            if(!move_in_move_array(move, moves2)) {
                return false;
            }
        }
    }
    return true;
}

bool presence_check(std::vector<std::vector<colour_t>> board, char row, char column, colour_t colour) {
    return board[row][column] == colour;
}

void test(bool result, std::string success, std::string failure) {
    if(result) {
        std::cout << success << "\n";
    } else {
        std::cout << failure << "\n";
    }
}

void wipe_board(std::vector<std::vector<colour_t>>* board) {
    //Erase all pieces from board
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            (*board)[i][j] = EMPTY;
        }
    }
}

void append_move(char x, char y, std::vector<move_t>* moves) {
    //Add move to move list
    Move move;
    move.x = x;
    move.y = y;
    moves->push_back(move);
}

int count_pieces(std::vector<std::vector<colour_t>>* board, colour_t colour) {
    int c = 0;
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            if((*board)[i][j] == colour) {
                c++;
            }
        }
    }
    return c;
}

void play_random_game() {
    std::vector<std::vector<colour_t>> board_copy = new_board();
    std::vector<move_t> whiteMoves, blackMoves;
    colour_t colour_turn_copy = BLACK;
    colour_t winner = NONE;
    move_t whiteMove, blackMove;
    int i, j;

    while(!game_has_ended(&board_copy)) {
        get_moves(&board_copy, &colour_turn_copy, &whiteMoves, &blackMoves);
        i = rand() % whiteMoves.size();
        j = rand() % blackMoves.size();
        whiteMove = whiteMoves[i];
        blackMove = blackMoves[j];
        make_move(&board_copy, whiteMove, blackMove, &colour_turn_copy, &winner);
    }
}

void test_board_1() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctWhiteMoves;
    colour_t colour_turn = WHITE;
    colour_t winner = NONE;
    bool test_success = false;

    std::cout << "Running test_board_1()\n";
    //white piece surrounded by 1 layer of black pieces
    std::vector<std::vector<colour_t>> board = new_board();
    wipe_board(&board);
    board[3][3] = WHITE;

    board[2][2] = BLACK;
    board[2][3] = BLACK;
    board[3][2] = BLACK;
    board[3][4] = BLACK;
    board[4][3] = BLACK;
    board[4][4] = BLACK;
    board[4][2] = BLACK;
    board[2][4] = BLACK;
    //Test 8 moves, all moves are correct
    append_move(1, 1, &correctWhiteMoves);
    append_move(1, 3, &correctWhiteMoves);
    append_move(3, 1, &correctWhiteMoves);
    append_move(3, 5, &correctWhiteMoves);
    append_move(5, 3, &correctWhiteMoves);
    append_move(5, 5, &correctWhiteMoves);
    append_move(5, 1, &correctWhiteMoves);
    append_move(1, 5, &correctWhiteMoves);
    
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);

    test(move_check(&whiteMoves, &correctWhiteMoves), "move_check(&whiteMoves, &correctWhiteMoves) succeeded", "move_check(&whiteMoves, &correctWhiteMoves) failed");
    //Test r
    make_move(&board, get_move(5, 3), get_null_move(), &colour_turn, &winner);
    test_success = board[5][3] == WHITE && board[4][3] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test r success", "Test r failure");
    colour_turn = WHITE;
    //Test l
    make_move(&board, get_move(1, 3), get_null_move(), &colour_turn, &winner);
    test_success = board[1][3] == WHITE && board[2][3] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test l success", "Test l failure");
    colour_turn = WHITE;
    //Test d
    make_move(&board, get_move(3, 5), get_null_move(), &colour_turn, &winner);
    test_success = board[3][5] == WHITE && board[3][4] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test d success", "Test d failure");
    colour_turn = WHITE;
    //Test u
    make_move(&board, get_move(3, 1), get_null_move(), &colour_turn, &winner);
    test_success = board[3][1] == WHITE && board[3][2] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test u success", "Test u failure");
    colour_turn = WHITE;
    //Test ul
    make_move(&board, get_move(1, 1), get_null_move(), &colour_turn, &winner);
    test_success = board[1][1] == WHITE && board[2][2] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test ul success", "Test ul failure");
    colour_turn = WHITE;
    //Test ur
    make_move(&board, get_move(5, 1), get_null_move(), &colour_turn, &winner);
    test_success = board[3][3] == WHITE && board[4][3] == WHITE && board[5][3] == WHITE;
    test(test_success, "Test ur success", "Test ur failure");
    colour_turn = WHITE;
    //Test dl
    make_move(&board, get_move(1, 5), get_null_move(), &colour_turn, &winner);
    test_success = board[1][5] == WHITE && board[2][4] == WHITE && board[3][3] == WHITE;
    test(test_success, "Test dl success", "Test dl failure");
    colour_turn = WHITE;
    //Test dr
    make_move(&board, get_move(5, 5), get_null_move(), &colour_turn, &winner);
    test_success = board[3][3] == WHITE && board[4][4] == WHITE && board[5][5] == WHITE;
    test(test_success, "Test dr success", "Test dr failure");
    colour_turn = WHITE;

    //Test the final board is as it should be
    test_success = count_pieces(&board, WHITE) == 17 && count_pieces(&board, BLACK) == 0;
    test(test_success, "Test final board success", "Test final board failure");

    print_game(&board, &colour_turn);
}

void test_board_2() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctWhiteMoves;
    colour_t colour_turn = WHITE;
    colour_t winner = NONE;
    bool test_success = false;

    std::cout << "Running test_board_2()\n";

    //white piece surrounded by 2 layer of black pieces
    std::vector<std::vector<colour_t>> board = new_board();
    wipe_board(&board);
    board[3][3] = WHITE;

    board[2][2] = BLACK;
    board[1][1] = BLACK;
    board[2][3] = BLACK;
    board[1][3] = BLACK;
    board[3][2] = BLACK;
    board[3][1] = BLACK;
    board[3][4] = BLACK;
    board[3][5] = BLACK;
    board[4][3] = BLACK;
    board[5][3] = BLACK;
    board[4][4] = BLACK;
    board[5][5] = BLACK;
    board[4][2] = BLACK;
    board[5][1] = BLACK;
    board[2][4] = BLACK;
    board[1][5] = BLACK;
    //Test 8 moves, all moves are correct
    append_move(0, 0, &correctWhiteMoves);
    append_move(0, 3, &correctWhiteMoves);
    append_move(3, 0, &correctWhiteMoves);
    append_move(3, 6, &correctWhiteMoves);
    append_move(6, 3, &correctWhiteMoves);
    append_move(6, 6, &correctWhiteMoves);
    append_move(6, 0, &correctWhiteMoves);
    append_move(0, 6, &correctWhiteMoves);
    
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(move_check(&whiteMoves, &correctWhiteMoves), "move_check(&whiteMoves, &correctWhiteMoves) succeeded", "move_check(&whiteMoves, &correctWhiteMoves) failed");


    make_move(&board, get_move(6, 3), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test l
    make_move(&board, get_move(0, 3), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test d
    make_move(&board, get_move(3, 6), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test u
    make_move(&board, get_move(3, 0), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test ul
    make_move(&board, get_move(0, 0), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test ur
    make_move(&board, get_move(6, 0), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test dl
    make_move(&board, get_move(0, 6), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;
    //Test dr
    make_move(&board, get_move(6, 6), get_null_move(), &colour_turn, &winner);
    colour_turn = WHITE;

    test_success = count_pieces(&board, WHITE) == 25 && count_pieces(&board, BLACK) == 0;
    test(test_success, "Test test_board_2() succeeded", "Test test_board_2() failed");
    print_game(&board, &colour_turn);
}

void test_board_3456() {
    //Test white piece flipping black piece at edge (done in test_board_2, no longer needed)
    //Test flipping at corners (done in test_board_2, no longer needed)

    //Test flipping along edges
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctBlackMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    bool test_success = false;

    std::cout << "Running test_board_3()\n";

    std::vector<std::vector<colour_t>> board = new_board();
    wipe_board(&board);
    board[7][0] = BLACK;
    board[7][1] = WHITE;
    board[7][2] = WHITE;
    board[7][3] = WHITE;

    append_move(7, 4, &correctBlackMoves);
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(move_check(&blackMoves, &correctBlackMoves), "move_check(&blackMoves, &correctBlackMoves) succeeded", "move_check(&blackMoves, &correctBlackMoves) failed");

    make_move(&board, get_move(7, 4), get_null_move(), &colour_turn, &winner);
    test_success = board[7][0] == BLACK && board[7][1] == BLACK && board[7][2] == BLACK && board[7][3] == BLACK && board[7][4] == BLACK && count_pieces(&board, WHITE) == 0 && count_pieces(&board, BLACK) == 5;
    test(test_success, "Test test_board_3() success", "Test test_board_3() failure");
    print_game(&board, &colour_turn);

    std::cout << "Running test_board_4()\n";
    blackMoves.clear();
    whiteMoves.clear();
    correctBlackMoves.clear();
    colour_turn = BLACK;
    winner = NONE;
    test_success = false;
    
    board = new_board();
    wipe_board(&board);
    board[0][4] = BLACK;
    board[0][1] = WHITE;
    board[0][2] = WHITE;
    board[0][3] = WHITE;

    append_move(0, 0, &correctBlackMoves);
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(move_check(&blackMoves, &correctBlackMoves), "move_check(&blackMoves, &correctBlackMoves) succeeded", "move_check(&blackMoves, &correctBlackMoves) failed");

    make_move(&board, get_move(0, 0), get_null_move(), &colour_turn, &winner);
    test_success = board[0][0] == BLACK && board[0][1] == BLACK && board[0][2] == BLACK && board[0][3] == BLACK && board[0][4] == BLACK && count_pieces(&board, WHITE) == 0 && count_pieces(&board, BLACK) == 5;
    test(test_success, "Test test_board_4() success", "Test test_board_4() failure");
    print_game(&board, &colour_turn);

    std::cout << "Running test_board_5()\n";
    blackMoves.clear();
    whiteMoves.clear();
    correctBlackMoves.clear();
    colour_turn = BLACK;
    winner = NONE;
    test_success = false;

    board = new_board();
    wipe_board(&board);
    board[7][7] = BLACK;
    board[6][7] = WHITE;
    board[5][7] = WHITE;
    board[4][7] = WHITE;

    append_move(3, 7, &correctBlackMoves);
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(move_check(&blackMoves, &correctBlackMoves), "move_check(&blackMoves, &correctBlackMoves) succeeded", "move_check(&blackMoves, &correctBlackMoves) failed");

    make_move(&board, get_move(3, 7), get_null_move(), &colour_turn, &winner);
    test_success = board[7][7] == BLACK && board[6][7] == BLACK && board[5][7] == BLACK && board[4][7] == BLACK && board[3][7] == BLACK && count_pieces(&board, WHITE) == 0 && count_pieces(&board, BLACK) == 5;
    test(test_success, "Test test_board_5() success", "Test test_board_5() failure");
    print_game(&board, &colour_turn);

    std::cout << "Running test_board_6()\n";
    blackMoves.clear();
    whiteMoves.clear();
    correctBlackMoves.clear();
    colour_turn = BLACK;
    winner = NONE;
    test_success = false;

    board = new_board();
    wipe_board(&board);
    board[7][0] = BLACK;
    board[7][1] = WHITE;
    board[7][2] = WHITE;
    board[7][3] = WHITE;
    board[7][4] = WHITE;
    board[7][5] = WHITE;
    board[7][6] = WHITE;

    append_move(7, 7, &correctBlackMoves);
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(move_check(&blackMoves, &correctBlackMoves), "move_check(&blackMoves, &correctBlackMoves) succeeded", "move_check(&blackMoves, &correctBlackMoves) failed");

    make_move(&board, get_move(7, 7), get_null_move(), &colour_turn, &winner);
    test_success = board[7][0] == BLACK && board[7][1] == BLACK && board[7][2] == BLACK && board[7][3] == BLACK && board[7][4] == BLACK && board[7][5] == BLACK && board[7][6] == BLACK && board[7][7] == BLACK && count_pieces(&board, WHITE) == 0 && count_pieces(&board, BLACK) == 8;
    test(test_success, "Test test_board_6() success", "Test test_board_6() failure");
    print_game(&board, &colour_turn);
}

void test_starting_position() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctBlackMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    bool test_success = false;

    std::cout << "Running test_starting_position()\n";
    std::vector<std::vector<colour_t>> board = new_board();
    append_move(3, 5, &correctBlackMoves);
    append_move(2, 4, &correctBlackMoves);
    append_move(5, 3, &correctBlackMoves);
    append_move(4, 2, &correctBlackMoves);
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    print_game(&board, &colour_turn);
    test(!game_has_ended(&board), "game_has_ended(&board) success", "game_has_ended(&board) failure");
    test(move_check(&blackMoves, &correctBlackMoves), "move_check(&blackMoves, &correctBlackMoves) succeeded", "move_check(&blackMoves, &correctBlackMoves) failed");
}

void test_get_winner() {
    //Starting position -> Draw
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctBlackMoves;
    colour_t colour_turn = BLACK;
    colour_t winner = NONE;
    bool test_success = false;
    std::vector<std::vector<colour_t>> board = new_board();

    std::cout << "Running test_get_winner()\n";
    //3 white pieces 2 black pieces -> White win
    test(determine_winner(&board) == DRAW, "determine_winner(&board) == DRAW success", "determine_winner(&board) == DRAW failure");
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(!game_has_ended(&board), "game_has_ended(&board) success", "game_has_ended(&board) failure");
    //3 black pieces 2 white pieces -> Black win
    board[0][0] = BLACK;
    test(determine_winner(&board) == BLACK, "determine_winner(&board) == BLACK success", "determine_winner(&board) == BLACK failure");
    //All white pieces - white win
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            board[i][j] = WHITE;
        }
    }
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(game_has_ended(&board), "game_has_ended(&board) success", "game_has_ended(&board) failure");
    test(determine_winner(&board) == WHITE, "determine_winner(&board) == BLACK success", "determine_winner(&board) == BLACK failure");


    //1 more black piece than white piece - black win
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            board[i][j] = i % 2 == 0 ? WHITE: BLACK;
        }
    }
    board[0][0] = BLACK;
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    test(game_has_ended(&board), "game_has_ended(&board) success", "game_has_ended(&board) failure");
    test(determine_winner(&board) == BLACK, "determine_winner(&board) == BLACK success", "determine_winner(&board) == BLACK failure");
}

void test_random_games() {
    std::cout << "Running test_random_games()\n";

    for(int i = 0; i < 1000; i++) {
        play_random_game();
    }
    test(true, "test_random_games() succeeded", "test_random_games() failed");
}

void test_player_cannot_move() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctWhiteMoves;
    colour_t colour_turn = WHITE;
    colour_t winner = NONE;
    bool test_success = false;
    std::vector<std::vector<colour_t>> board = new_board();
    wipe_board(&board);

    std::cout << "Running test_player_cannot_move()\n";

    board[1][1] = WHITE;
    board[0][0] = BLACK;
    
    //Black can move, white cannot
    test(!game_has_ended(&board), "!game_has_ended(&board) success", "!game_has_ended(&board) failure");
    get_moves(&board, &colour_turn, &whiteMoves, &blackMoves);
    append_move(-1, -1, &correctWhiteMoves);
    test(move_check(&whiteMoves, &correctWhiteMoves), "move_check(&whiteMoves, &correctWhiteMoves) success", "move_check(&whiteMoves, &correctWhiteMoves) failure");
    make_move(&board, get_null_move(), get_null_move(), &colour_turn, &winner);
    test(colour_turn == BLACK, "colour_turn == BLACK success", "colour_turn == BLACK failure");
}

void test_both_player_cannot_move() {
    std::vector<move_t> blackMoves;
    std::vector<move_t> whiteMoves;
    std::vector<move_t> correctWhiteMoves;
    colour_t colour_turn = WHITE;
    colour_t winner = NONE;
    bool test_success = false;
    std::vector<std::vector<colour_t>> board = new_board();
    wipe_board(&board);

    std::cout << "Running test_both_player_cannot_move()\n";

    test(game_has_ended(&board), "game_has_ended(&board) success", "game_has_ended(&board) failure");
}

void run_tests() {
    test_board_1();
    test_board_2();
    test_board_3456();
    test_get_winner();
    test_starting_position();
    test_random_games();
    test_player_cannot_move();
    test_both_player_cannot_move();
}*/