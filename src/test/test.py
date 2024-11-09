"""
some test to verify that various stuff is actually behaving as expected
"""
from src.environment.board import OthelloBoard
from src.test.test_utils import *
import string, random

def run_all_tests():
    print("Running tests..")

    print("check default board at beginning..")
    try_good_board_config(8,8)
    try_good_board_config(10,10)
    try_good_board_config(6,6)

    print("check some bad board configs..")
    try_bad_board_config(5, 5)
    try_bad_board_config(8, 9)
    try_bad_board_config(7, 8)
    try_bad_board_config(5, 6)

    print("trying out some moves..")
    board = OthelloBoard()
    try_valid_move(board, "D3", 1)
    try_valid_move(board, "C3", 0)
    try_valid_move(board, "C4", 1)
    board.print_board()

    black_player_valid_moves = ['E3', 'C5']
    print(f"Player black is next and can do these moves: {black_player_valid_moves}")
    for move in black_player_valid_moves:
        boardTemp = board.copy()
        try_valid_move(boardTemp, move, 0)

    all_moves = [f"{letter}{number}" for letter in string.ascii_uppercase[:8] for number in range(1, 9)]
    print(f"black player should not be able to do any of the other 62 moves..{[move for move in all_moves if move not in black_player_valid_moves]}")
    for move in all_moves:
        if move not in black_player_valid_moves:
            boardTemp = board.copy()
            try_invalid_move(boardTemp, move, 0)
    print("let black play..")
    try_valid_move(board, "E3", 0)
    print("Field now for white player:")
    board.print_board()

    white_player_valid_moves = ['B2', 'C2', 'D2', 'E2', 'F2', 'F3', 'F4', 'F5', 'F6']
    print(f"white player is next and can do these moves: {white_player_valid_moves}")
    for move in white_player_valid_moves:
        boardTemp = board.copy()
        try_valid_move(boardTemp, move, 1)

    print(f"white player should not be able to do any of the other moves..{[move for move in all_moves if move not in white_player_valid_moves]}")
    for move in all_moves:
        if move not in white_player_valid_moves:
            boardTemp = board.copy()
            try_invalid_move(boardTemp, move, 1)

    print("finish game randomly..")
    current_player = 1
    current_valid_moves = board.get_valid_moves(current_player)

    while len(current_valid_moves) > 0:
        current_player = board.make_move(random.choice(current_valid_moves), current_player)
        current_valid_moves = board.get_valid_moves(current_player)

    board.print_board()

    print("\n\nAll tests passed!")

if __name__ == "__main__":
    run_all_tests()