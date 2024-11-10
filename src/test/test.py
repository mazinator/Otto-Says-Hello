"""
some test to verify that various stuff is actually behaving as expected
"""
from src.agents.simple_agents import RandomAgent, MinAgent, MaxAgent
from src.environment.board import OthelloBoard
from src.utils.results_printer import store_results
import string, random, time


def run_all_tests(run_excessive=False, print_incomplete_boards=False, print_boards=False):
    print("Running tests..")

    print("check default board at beginning..")
    try_good_board_config(8, 8)
    try_good_board_config(10, 10)
    try_good_board_config(6, 6)

    print("check some bad board configs..")
    try_bad_board_config(5, 5)
    try_bad_board_config(8, 9)
    try_bad_board_config(7, 8)
    try_bad_board_config(5, 6)

    print("trying out some moves..")
    board = OthelloBoard(rows=10, cols=10)
    board.print_board()
    try_valid_move(board, "E4", 0)
    board.print_board()
    try_valid_move(board, "F4", 1)
    board.print_board()
    try_valid_move(board, "G5", 0)
    board.print_board()

    white_player_valid_moves = ['D4', 'D6', 'H4', 'H6']
    print(f"Player white is next and can do these moves: {white_player_valid_moves}")
    for move in white_player_valid_moves:
        boardTemp = board.copy()
        try_valid_move(boardTemp, move, 1)

    all_moves = [f"{letter}{number}" for letter in string.ascii_uppercase[:8] for number in range(1, 9)]
    print(
        f"white player should not be able to do any of the other moves..{[move for move in all_moves if move not in white_player_valid_moves]}")
    for move in all_moves:
        if move not in white_player_valid_moves:
            boardTemp = board.copy()
            try_invalid_move(boardTemp, move, 1)
    print("White making a move..")
    try_valid_move(board, "D4", 1)
    print("let black play..")
    print("Field now for black player:")
    board.print_board()

    black_player_valid_moves = ['F3', 'E3', 'D5', 'F7', 'D3', 'E7', 'G6']
    print(f"black player is next and can do these moves: {black_player_valid_moves}")
    for move in black_player_valid_moves:
        boardTemp = board.copy()
        try_valid_move(boardTemp, move, 0)

    print(
        f"white player should not be able to do any of the other moves..{[move for move in all_moves if move not in black_player_valid_moves]}")
    for move in all_moves:
        if move not in black_player_valid_moves:
            boardTemp = board.copy()
            try_invalid_move(boardTemp, move, 0)

    print("finish game randomly..")
    current_player = 0
    current_valid_moves = board.get_valid_moves(current_player)

    while len(current_valid_moves) > 0:
        current_player, current_board, current_valid_moves, winner = board.make_move(random.choice(current_valid_moves),
                                                                                     current_player)

    board.print_board()

    print("printing a few bigger boards..")
    try_good_board_config(10, 10)
    try_good_board_config(row=20, col=20)
    try_good_board_config(row=20, col=26)

    print("trying board too big..")
    try_bad_board_config(27, 25)
    try_bad_board_config(27, 29)
    try_bad_board_config(27, 29)

    print("\nplaying a few random games on the biggest board possible .. ")
    if run_excessive:
        print("CAUTION; RUNNING EXZESSIVE TESTS!")
        for i in range(5):
            board = OthelloBoard(rows=26, cols=26)
            current_player = 0
            current_valid_moves = board.get_valid_moves(current_player)
            while len(current_valid_moves) > 0:
                current_player, current_board, current_valid_moves, winner = board.make_move(
                    random.choice(current_valid_moves),
                    current_player)

    if run_excessive:
        print("\nchecking if central limit theorem kicks in when playing a shitload of rounds..")
        determine_average_winner(50000, print_boards=print_boards,
                                 print_only_uncomplete_boards=print_incomplete_boards)
    else:
        determine_average_winner(episodes=1000, print_boards=print_boards,
                                 print_only_uncomplete_boards=print_incomplete_boards)

    print("\nPlaying RandomAgent vs. MinAgent on 8x8 board..")
    agent_vs_agent(OthelloBoard(rows=8, cols=8), black=RandomAgent(), white=MinAgent(), print_boards=print_boards,
                   delay=0)

    print("\nPlaying RandomAgent vs. MaxAgent..")
    agent_vs_agent(OthelloBoard(), black=RandomAgent(), white=MaxAgent(), print_boards=print_boards, delay=0)

    print("\nPlaying MinAgent vs. MaxAgent..")
    agent_vs_agent(OthelloBoard(), black=MinAgent(), white=MaxAgent(), print_boards=print_boards, delay=0)

    print("\nPlaying MinAgent vs. MaxAgent with both having epsilon 0.05 to introduce some level of randomness..")
    agent_vs_agent(OthelloBoard(), black=MinAgent(0.05), white=MaxAgent(0.05), print_boards=print_boards, delay=0,
                   episodes=10000)

    print("\n\nALL TESTS PASSED!")


def agent_vs_agent(board, black, white, print_boards=False, episodes=50, delay=0.1, print_only_final_result=False,
                   print_incomplete_boards=False):
    print(f"Starting to play {black.__class__.__name__} vs. {white.__class__.__name__} with {episodes} episodes.")
    black_wins, white_wins, ties = 0, 0, 0

    for episode in range(episodes):
        if episode % 100 == 0 and episode != 0:
            print(f"playing Episode {episode} of {episodes}..")
        board.reset_board()
        current_player, winner = 0, 0  # 0 for Black, 1 for White

        while True:
            # Get valid moves for the current player
            valid_moves = board.get_valid_moves(current_player)

            # Check if there are no valid moves for either player, ending the game
            if not valid_moves and not board.get_valid_moves(1 - current_player):
                winner = board.check_winner()
                break

            # If no valid moves for the current player, switch to the other player
            if not valid_moves:
                current_player = 1 - current_player
                continue

            # Choose the agent based on the current player
            agent = black if current_player == 0 else white
            move = agent.get_action(board, current_player)

            # Make the move
            next_player, _, _, winner = board.make_move(move, current_player)

            # Print board if requested
            if print_boards and not print_only_final_result:
                print(f"Episode {episode + 1}, {['Black', 'White'][current_player]}'s move:")
                board.print_board()
                time.sleep(delay)

            # If the game is over, record the winner
            if winner < 0:
                if winner == -1:
                    black_wins += 1
                    print("Black wins!") if print_boards else None
                elif winner == -2:
                    white_wins += 1
                    print("White wins!") if print_boards else None
                else:
                    ties += 1
                    print("It's a tie!") if print_boards else None
                break

            # Check for consecutive moves if the next player has no valid moves
            current_player = next_player

        if print_boards and print_only_final_result:
            board.print_board()
            time.sleep(delay)

        if print_boards and print_incomplete_boards and board.is_incomplete():
            print("Incomplete board detected!")
            board.print_board()

    # Print results after all episodes
    print(f"Results after {episodes} episodes: Black wins: {black_wins}, White wins: {white_wins}, Ties: {ties}")
    store_results(black_wins=black_wins, white_wins=white_wins, draws=ties, episodes=episodes,
                  black_agent=black.__class__.__name__, white_agent=white.__class__.__name__)


def try_invalid_move(board, move, player):
    try:
        board.make_move(move, player)
        assert False, print(f"Move {move} is NOT valid")
    except ValueError:
        print(f"Test passed, move {move} is valid")


def try_valid_move(board, move, player):
    try:
        board.make_move(move, player)
        print(f"Test passed, move {move} is valid")
    except ValueError:
        assert False, print(f"Move {move} should be valid!")


def try_bad_board_config(col, row):
    try:
        boardNok = OthelloBoard(cols=col, rows=row)
        assert False, print(f"board with col={col}, row={row} should NOT be possible")
    except ValueError:
        print(f"Test passed")


def try_good_board_config(col, row):
    try:
        boardOk = OthelloBoard(cols=col, rows=row)
        print("Test passed")
    except ValueError:
        assert False, print(f"board with col={col}, row={row} should be possible")


def determine_average_winner(episodes=1000, rows=6, cols=6, print_boards=False, print_only_uncomplete_boards=False):
    print("using alternating beginner (black/white) to remove issues from who starts in random games..")
    white_wins = 0
    black_wins = 0
    draws = 0
    for i in range(episodes):
        if i % 5000 == 0:
            print(f"Episode {i} in CLT checker of {episodes}")
        res = play_random_game(rows, cols, print_boards, current_player=0 if i % 2 == 0 else 1,
                               print_only_incomplete_boards=print_only_uncomplete_boards)

        if res == -1:
            black_wins += 1
        elif res == -2:
            white_wins += 1
        else:
            draws += 1
    print(f"black wins: {black_wins}. white wins: {white_wins}. draws: {draws}")
    print(f"black wins pct. = {black_wins / (black_wins + white_wins)}")

    win_pct = black_wins / (episodes - draws)
    if episodes >= 10000:
        assert win_pct > 0.45 and win_pct < 0.55, f"weird to have such a discreptany with {episodes} episodes"


def play_random_game(cols=6, rows=6, print_board=False, current_player=0, print_only_incomplete_boards=False):
    """
    returns the winner
    """

    board = OthelloBoard(rows=rows, cols=cols, first_player=current_player)
    current_valid_moves = board.get_valid_moves(current_player)
    while len(current_valid_moves) > 0:
        current_player, current_board, current_valid_moves, winner = board.make_move(random.choice(current_valid_moves),
                                                                                     current_player)
    if print_board:
        result = "Black wins" if winner == -1 else "White wins" if winner == -2 else "Tie"
        print(f"\n The result: {result}")
        board.print_board()

    if print_only_incomplete_boards and board.is_incomplete():
        print("\nIncomplete board detected!")
        result = "Black wins" if winner == -1 else "White wins" if winner == -2 else "Tie"
        print(f"The result: {result}")
        board.print_board()

    return winner


if __name__ == "__main__":
    run_all_tests(run_excessive=False, print_boards=False, print_incomplete_boards=False)
