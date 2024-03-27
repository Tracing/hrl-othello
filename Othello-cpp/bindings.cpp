#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "ai.h"
#include "constants.h"
#include "engine.h"
#include "objects.h"
#include "io.h"

move_t py_get_move(int x, int y) {
    return get_move((char) x, (char) y);
}

OthelloGame* py_new_OthelloGame() {
    OthelloGame* game = new OthelloGame();
    return game;
}

void py_destroy_OthelloGame(OthelloGame* game) {
    delete game;
}

move_t get_minimax_move_1(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, int depth, unsigned int seed) {
    std::vector<float> goal;
    return get_minimax_move(board, &goal, false, colour_turn, depth, seed);
}

move_t get_minimax_move_2(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, colour_t* colour_turn, int depth, unsigned int seed) {
    return get_minimax_move(board, goal, true, colour_turn, depth, seed);
}

PYBIND11_MODULE(othello, m) {
    m.def("get_null_move", &get_null_move);
    m.def("get_move", &py_get_move);
    m.def("play_random_game", &play_random_game);
    m.def("print_game", &print_game);
    m.def("get_minimax_move", &get_minimax_move_1);
    m.def("get_minimax_goal_move", &get_minimax_move_2);
    m.def("get_mcts_move", &get_mcts_move);
    m.def("new_OthelloGame", &py_new_OthelloGame, pybind11::return_value_policy::reference);
    m.def("free_OthelloGame", &py_destroy_OthelloGame);

    pybind11::class_<Move>(m, "Move")
        .def(pybind11::init<>())
        .def_readwrite("x", &Move::x)
        .def_readwrite("y", &Move::y);

    pybind11::class_<OthelloGame>(m, "OthelloGame")
        .def(pybind11::init<>())
        .def_readwrite("board", &OthelloGame::board)
        .def_readwrite("colour_turn", &OthelloGame::colour_turn)
        .def_readwrite("winner", &OthelloGame::winner)
        .def_readwrite("whiteMoves", &OthelloGame::whiteMoves)
        .def_readwrite("blackMoves", &OthelloGame::blackMoves)
        .def("game_has_ended", &OthelloGame::game_has_ended)
        .def("get_board", &OthelloGame::get_board, pybind11::return_value_policy::reference)
        .def("at_square", &OthelloGame::at_square)
        .def("set_square", &OthelloGame::set_square)
        .def("reset", &OthelloGame::reset)
        .def("do_move", &OthelloGame::do_move)
        .def("copy", &OthelloGame::copy);
}
