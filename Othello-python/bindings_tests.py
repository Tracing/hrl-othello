import othello
import random

def play_random_game():
    game = othello.new_OthelloGame()
    while not game.game_has_ended():
        game.do_move(random.choice(game.whiteMoves), random.choice(game.blackMoves))

def play_random_games():
    for _ in range(10000):
        play_random_game()

play_random_games()