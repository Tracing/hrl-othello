#include "ai.h"
#include "constants.h"
#include "engine.h"
#include "objects.h"
#include "interface.h"
#include "tests.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    time_t t;
    srand(1);
    /*if(N == 8) {
        run_tests();
    }*/
    run_game(AI_MCTS, AI_MCTS, 30, 30, 100000, 100000, sqrt(2), sqrt(2));
    return 0;
}
