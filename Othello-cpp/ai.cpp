#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include "ai.h"
#include "constants.h"
#include "engine.h"

float alphabeta(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, bool use_goal, colour_t* colour_turn, colour_t for_colour, int depth, float alpha, float beta) {
    float value;
    std::vector<std::vector<colour_t>> new_board;
    colour_t new_colour_turn;
    colour_t new_winner;
    std::vector<move_t> whiteMoves, blackMoves;
    std::vector<move_t>* moves;
    move_t whiteMove, blackMove;

    if(depth == 0 || game_has_ended(board, *colour_turn)) {
        return heuristic_fn(board, goal, use_goal, *colour_turn, for_colour);
    }
    if(*colour_turn == for_colour) {
        value = -1000;
        get_moves(board, colour_turn, &whiteMoves, &blackMoves);
        if(*colour_turn == WHITE) {
            moves = &whiteMoves;
        } else {
            moves = &blackMoves;
        }
        for(int i = 0; i < moves->size(); i++) {
            if(*colour_turn == WHITE) {
                whiteMove = whiteMoves[i];
                blackMove = blackMoves[0];
            } else {
                whiteMove = whiteMoves[0];
                blackMove = blackMoves[i];
            }
            new_board = copy_board(board);
            new_colour_turn = *colour_turn;
            new_winner = NONE;
            make_move(&new_board, whiteMove, blackMove, &new_colour_turn, &new_winner);
            value = std::max(value, alphabeta(&new_board, goal, use_goal, &new_colour_turn, for_colour, depth - 1, alpha, beta));
            if(value >= beta) {
                break;
            }
            alpha = std::max(alpha, value);
        }
        return value;
    } else {
        value = 1000;
        get_moves(board, colour_turn, &whiteMoves, &blackMoves);
        if(*colour_turn == WHITE) {
            moves = &whiteMoves;
        } else {
            moves = &blackMoves;
        }
        for(int i = 0; i < moves->size(); i++) {
            if(*colour_turn == WHITE) {
                whiteMove = whiteMoves[i];
                blackMove = blackMoves[0];
            } else {
                whiteMove = whiteMoves[0];
                blackMove = blackMoves[i];
            }
            new_board = copy_board(board);
            new_colour_turn = *colour_turn;
            new_winner = NONE;
            make_move(&new_board, whiteMove, blackMove, &new_colour_turn, &new_winner);
            value = std::min(value, alphabeta(&new_board, goal, use_goal, &new_colour_turn, for_colour, depth - 1, alpha, beta));
            if(value <= alpha) {
                break;
            }
            alpha = std::min(alpha, value);
        }
        return value;
    }
}

move_t get_minimax_move(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, bool use_goal, colour_t* colour_turn, int depth, unsigned int seed) {
    float best_score = -1000;
    float value;
    move_t bestMove;
    std::vector<std::vector<colour_t>> new_board;
    colour_t new_colour_turn;
    colour_t new_winner;
    colour_t for_colour = *colour_turn;
    std::vector<move_t> whiteMoves, blackMoves;
    std::vector<move_t>* moves;
    move_t whiteMove, blackMove;

    srand(seed);
    get_moves(board, colour_turn, &whiteMoves, &blackMoves);

    if(*colour_turn == WHITE) {
        moves = &whiteMoves;
    } else {
        moves = &blackMoves;
    }
    for(int i = 0; i < moves->size(); i++) {
        if(*colour_turn == WHITE) {
            whiteMove = whiteMoves[i];
            blackMove = blackMoves[0];
        } else {
            whiteMove = whiteMoves[0];
            blackMove = blackMoves[i];
        }
        new_board = copy_board(board);
        new_colour_turn = *colour_turn;
        new_winner = NONE;
        make_move(&new_board, whiteMove, blackMove, &new_colour_turn, &new_winner);
        value = alphabeta(&new_board, goal, use_goal, &new_colour_turn, for_colour, depth - 1, -10000, 10000);
        if(value > best_score) {
            best_score = value;
            if(*colour_turn == WHITE) {
                bestMove = whiteMove;
            } else {
                bestMove = blackMove;
            }
        }
    }
    return bestMove;
}

mcts_tree_t new_mcts_node(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, mcts_tree_t* parent) {
    mcts_tree_t node;
    std::vector<move_t> blackMoves, whiteMoves;
    get_moves(board, colour_turn, &whiteMoves, &blackMoves);
    node.colour_turn = *colour_turn;
    node.board = copy_board(board);
    node.visits = 0;
    node.score = 0;
    node.blackMoves = blackMoves;
    node.whiteMoves = whiteMoves;
    node.winner = NONE;
    node.parent = parent;
    node.terminal = false;
    return node;
}

bool fully_expanded(mcts_tree_t* node) {
    if(node->colour_turn == WHITE) {
        return node->children.size() == node->whiteMoves.size();
    } else {
        return node->children.size() == node->blackMoves.size();
    } 
           
}

bool is_leaf(mcts_tree_t* node) {
    return node->terminal || (!fully_expanded(node));
}

mcts_tree_t* best_uct(mcts_tree_t* node, float C) {
    float score;
    float best_score = FLT_MIN;
    mcts_tree_t* child;
    mcts_tree_t* best_child = nullptr;

    if(node->children.size() == 0) {
        best_child = node;
    } else if(node->children.size() == 1) {
        best_child = &node->children[0];
    } else {
        for(int i = 0; i < node->children.size(); i++) {
            child = &node->children[i];
            assert(child != nullptr);

            if(child->visits == 0) {
                best_child = child;
                break;
            }
            if(child->colour_turn == node->colour_turn) {
                score = child->score / child->visits + C * sqrt(log(node->visits) / child->visits);
            } else {
                score = (1 - (child->score / child->visits)) + C * sqrt(log(node->visits) / child->visits);
            }
            if(score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
    }
    assert(best_child != nullptr);
    return best_child;
}

mcts_tree_t* traverse(mcts_tree_t* root, float C) {
    mcts_tree_t* current = root;
    while(!is_leaf(current)) {
        current = best_uct(current, C);
    }
    return current;
}

mcts_tree_t* expansion(mcts_tree_t* node) {
    std::vector<move_t>* moves = node->colour_turn == WHITE ? &node->whiteMoves : &node->blackMoves;
    move_t* new_move = &(*moves)[node->children.size()];
    mcts_tree_t* new_node;
    if(!node->terminal) {
        node->children.push_back(new_mcts_node(&node->board, &node->colour_turn, node));
        new_node = &node->children[node->children.size() - 1];
        if(new_node->colour_turn == WHITE) {
            make_move(&new_node->board, *new_move, node->blackMoves[0], &new_node->colour_turn, &new_node->winner);
        } else {
            make_move(&new_node->board, node->whiteMoves[0], *new_move, &new_node->colour_turn, &new_node->winner);
        }
        if(game_has_ended(&new_node->board, *(&new_node->colour_turn))) {
            new_node->terminal = true;
        } else {
            get_moves(&new_node->board, &new_node->colour_turn, &new_node->whiteMoves, &new_node->blackMoves);
        }
    } else {
        new_node = node;
    }
    return new_node;
}

float simulation(mcts_tree_t* node, colour_t player) {
    float score;
    colour_t winner;
    if(node->winner == NONE) {
        winner = play_random_game(&node->board, &node->colour_turn);
    } else {
        winner = node->winner;
    }
    if(winner == player) {
        score = 1;
    } else if(winner == DRAW) {
        score = 0.5;
    } else {
        score = 0.0;
    }
    return score;
}

void backpropogation(mcts_tree_t* node, float score, colour_t player) {
    while(node->parent != nullptr) {
        node->visits += 1;
        node->score += score;
        node = node->parent;
    }
    node->visits += 1;
    node->score += score;
}

move_t get_mcts_best_move(mcts_tree_t* root) {
    float best_score = -1;
    float score;
    move_t* best_move;
    mcts_tree_t* child;
    for(int i = 0; i < root->children.size(); i++) {
        child = &root->children[i];
        if(child->visits == 0) {
            score = 0.0;
        } else {
            score = child->score / child->visits;
        }
        if(score > best_score) {
            best_move = root->colour_turn == WHITE ? &root->whiteMoves[i] : &root->blackMoves[i];
            best_score = score;
        }
    }
    return *best_move;
}

move_t get_mcts_move(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn, int n, float C, unsigned int seed) {
    mcts_tree_t root = new_mcts_node(board, colour_turn, nullptr);
    mcts_tree_t* current;
    colour_t player = *colour_turn;
    float score;
    srand(seed);
    for(int i = 0; i < n; i++) {
        //Selection
        current = traverse(&root, C);
        //Expansion
        current = expansion(current);
        //Simulation
        score = simulation(current, player);
        //Backpropogation
        backpropogation(current, score, player);
    }
    return get_mcts_best_move(&root);
}

float _get_coin_parity_score(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    float a = 0.0;
    float b = 0.0;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] == for_colour) {
                a += 1;
            } else if((*board)[i][j] == EMPTY || (*board)[i][j] == BLOCKING) {
                ;
            } else {
                b += 1;
            }
        }
    }

    if((a + b) < 1e-2) {
        return 0.0;
    } else {
        return a / (a + b);
    }
}

bool _is_diagonal_filled(std::vector<std::vector<colour_t>>* board, int i, int j, int di, int dj) {
    int l1 = 0;
    int l2 = 0;
    bool filled;

    while(i < N && j < 0 && i >= 0 && j >= 0) {
        l1 += abs((*board)[i][j]);
        l2 += 1;
        i += di;
        j += dj;
    }
    if(l1 == 0 && l2 == 0) {
        filled = true;
    } else {
        filled = l1 == l2;
    }
    return filled;
}

int _n_diagonals_filled(std::vector<std::vector<colour_t>>* board, int i, int j) {
    int score1 = 0;
    int score2 = 0;

    score1 += _is_diagonal_filled(board, i, j, 1, 1) ? 1 : 0;
    score1 += _is_diagonal_filled(board, i, j, -1, -1) ? 1 : 0;
    score1 = score1 > 1 ? 1 : 0;

    score2 += _is_diagonal_filled(board, i, j, 1, -1) ? 1 : 0;
    score2 += _is_diagonal_filled(board, i, j, -1, 1) ? 1 : 0;
    score2 = score2 > 1 ? 1 : 0;

    return score1 + score2;
}

bool _is_row_filled(std::vector<std::vector<colour_t>>* board, int i) {
    int x = 0;
    for(int j = 0; j < N; j++) {
        x += abs((*board)[i][j]);
    }
    return x == N;
}

bool _is_column_filled(std::vector<std::vector<colour_t>>* board, int i) {
    int x = 0;
    for(int j = 0; j < N; j++) {
        x += abs((*board)[j][i]);
    }
    return x == N;
}

std::vector<std::vector<int>> _get_n_lanes(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    std::vector<std::vector<int>> lanes_owned;
    colour_t other_colour = for_colour == WHITE ? BLACK : WHITE;
    int n_lanes = 0;

    for(int i = 0; i < N; i++) {
        lanes_owned.push_back(std::vector<int>());
        for(int j = 0; j < N; j++) {
            n_lanes += _is_row_filled(board, i) ? 1 : 0;
            n_lanes += _is_column_filled(board, i) ? 1 : 0;
            n_lanes += _n_diagonals_filled(board, i, j);
            lanes_owned[i].push_back(n_lanes);
        }
    }

    lanes_owned[0][0] = 4;
    lanes_owned[7][7] = 4;
    lanes_owned[0][7] = 4;
    lanes_owned[7][0]= 4;
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] == for_colour) {
                ;
            } else if((*board)[i][j] == other_colour) {
                lanes_owned[i][j] = -lanes_owned[i][j];
            } else {
                lanes_owned[i][j] = 0;
            }
        }
    }

    return lanes_owned;
}

float _get_stability_score(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    float a = 0.0;
    float b = 0.0;
    std::vector<std::vector<int>> lanes_owned = _get_n_lanes(board, for_colour);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] == for_colour) {
                a += lanes_owned[i][j] == 4 ? 1.0: 0.0;
            } else if((*board)[i][j] == EMPTY || (*board)[i][j] == BLOCKING) {
                ;
            } else {
                b += lanes_owned[i][j] == -4 ? 1.0: 0.0;
            }
        }
    }

    if((a + b) < 1e-2) {
        return 0.0;
    } else {
        return a / (a + b);
    }
}

float _get_lanes_owned_score(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    float a = 0.0;
    float b = 0.0;
    std::vector<std::vector<int>> lanes_owned = _get_n_lanes(board, for_colour);

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if((*board)[i][j] == for_colour) {
                a += lanes_owned[i][j];
            } else if((*board)[i][j] == EMPTY || (*board)[i][j] == BLOCKING) {
                ;
            } else {
                b += lanes_owned[i][j];
            }
        }
    }

    if((a + b) < 1e-2) {
        return 0.0;
    } else {
        return a / (a + b);
    }
}

float _get_edges_score(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    float a = 0.0;
    float b = 0.0;

    for(int i = 2; i < 6; i++) {
        if((*board)[i][7] == for_colour) {
            a += 1;
        } else if((*board)[i][7] == EMPTY || (*board)[i][7] == BLOCKING) {
            ;
        } else {
            b += 1;
        }

        if((*board)[7][i] == for_colour) {
            a += 1;
        } else if((*board)[7][i] == EMPTY || (*board)[7][i] == BLOCKING) {
            ;
        } else {
            b += 1;
        }

        if((*board)[0][i] == for_colour) {
            a += 1;
        } else if((*board)[0][i] == EMPTY || (*board)[0][i] == BLOCKING) {
            ;
        } else {
            b += 1;
        }

        if((*board)[i][0] == for_colour) {
            a += 1;
        } else if((*board)[i][0] == EMPTY || (*board)[i][0] == BLOCKING) {
            ;
        } else {
            b += 1;
        }
    }

    if((a + b) < 1e-2) {
        return 0.0;
    } else {
        return a / (a + b);
    }
}

float _get_corner_score(std::vector<std::vector<colour_t>>* board, colour_t for_colour) {
    float a = 0.0;
    float b = 0.0;

    if((*board)[7][7] == for_colour) {
        a += 1;
    } else if((*board)[7][7] == EMPTY || (*board)[7][7] == BLOCKING) {
        ;
    } else {
        b += 1;
    }

    if((*board)[7][0] == for_colour) {
        a += 1;
    } else if((*board)[7][0] == EMPTY || (*board)[7][0] == BLOCKING) {
        ;
    } else {
        b += 1;
    }

    if((*board)[0][7] == for_colour) {
        a += 1;
    } else if((*board)[0][7] == EMPTY || (*board)[0][7] == BLOCKING) {
        ;
    } else {
        b += 1;
    }

    if((*board)[0][0] == for_colour) {
        a += 1;
    } else if((*board)[0][0] == EMPTY || (*board)[0][0] == BLOCKING) {
        ;
    } else {
        b += 1;
    }

    if((a + b) < 1e-2) {
        return 0.0;
    } else {
        return a / (a + b);
    }
}

float heuristic_fn(std::vector<std::vector<colour_t>>* board, std::vector<float>* goal, bool use_goal, colour_t colour_turn, colour_t for_colour) {
    float score = ((float) rand()) / ((float) RAND_MAX);
    colour_t winner = NONE;
    colour_t against_colour = for_colour == WHITE ? BLACK : WHITE;
    if(use_goal) {
        score = 0;
        score += (*goal)[0] * _get_coin_parity_score(board, for_colour);
        score += (*goal)[1] * _get_stability_score(board, for_colour);
        score += (*goal)[2] * _get_lanes_owned_score(board, for_colour);
        score += (*goal)[3] * _get_corner_score(board, for_colour);
        score += (*goal)[4] * _get_edges_score(board, for_colour);
        score /= ((*goal)[0] + (*goal)[1] + (*goal)[2] + (*goal)[3] + (*goal)[4]);
        score = (score - 0.5) * 2;
        score += ((float) rand()) / ((float) RAND_MAX) * 1e-5;
    } else {
        if(game_has_ended(board, colour_turn)) {
            winner = determine_winner(board);
            if(winner == for_colour) {
                score = 100;
            } else if(winner == DRAW) {
                score = 0;
            } else {
                score = -100;
            }
        } else {
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                    if((*board)[i][j] == for_colour) {
                        score += 1;
                    } else if((*board)[i][j] == EMPTY || (*board)[i][j] == BLOCKING) {
                        ;
                    } else {
                        score -= 1;
                    }
                }
            }
        }
    }
    return score;
}

colour_t play_random_game(std::vector<std::vector<colour_t>>* board, colour_t* colour_turn) {
    std::vector<std::vector<colour_t>> board_copy = copy_board(board);
    std::vector<move_t> whiteMoves, blackMoves;
    colour_t colour_turn_copy = *colour_turn;
    colour_t winner = NONE;
    move_t whiteMove, blackMove;
    int i, j;

    while(!game_has_ended(&board_copy, colour_turn_copy)) {
        get_moves(&board_copy, &colour_turn_copy, &whiteMoves, &blackMoves);
        i = rand() % whiteMoves.size();
        j = rand() % blackMoves.size();
        whiteMove = whiteMoves[i];
        blackMove = blackMoves[j];
        make_move(&board_copy, whiteMove, blackMove, &colour_turn_copy, &winner);
    }
    return winner;
}