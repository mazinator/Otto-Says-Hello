from src.environment.board import OthelloBoard
def try_invalid_move(board, move, player):
    try:
        board.make_move(move, player)
        assert False, print(f"Move {move} is NOT valid")
    except ValueError:
        assert True, print(f"Test passed, move {move} is valid")


def try_valid_move(board, move, player):
    try:
        board.make_move(move, player)
        assert True, print(f"Test passed, move {move} is valid")
    except ValueError:
        assert False, print(f"Move {move} should be valid!")


def try_bad_board_config(col, row):
    try:
        boardNok = OthelloBoard(cols=col, rows=row)
        assert False, print(f"board with col={col}, row={row} should NOT be possible")
    except ValueError:
        assert True, print(f"Test passed")


def try_good_board_config(col, row):
    try:
        boardOk = OthelloBoard(cols=col, rows=row)
        assert True, print("Test passed")
    except ValueError:
        assert False, print(f"board with col={col}, row={row} should be possible")