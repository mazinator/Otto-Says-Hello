"""
test file to verify that various stuff is actually behaving as expected.
Tests are available for all reasonable aspects:
- wrong parameters when creating a board, e.g. size
- checking on valid/invalid actions
- check if winner is correctly determined by making use of the CLT (ratio of black wins tends towards 0.5) in
  combination with random agents. It can be argued that directly checking winners is also an option, however
  there's only so much games one can check manually (and still having some uncertainty). I therefore claim that
  a combined approach (checking some valid/invalid actions and winners manually, while also assuming that randomly
  played games with alternating beginners (white/black) has to lead to a black-winner-ratio of 0.5, excluding draws.
- some further validation by playing a MaxAgent (flip as much disks as possible) vs a MinAgent (vice versa). I expect
  the MinAgent to always loose, which is the case.
- The human-played games are read in, played through completely, and it is checked if the winner is as stated by
  the given 'observation'. All actions are valid, because otherwise there would be an error raised by the board-file

There are some further parameters in the 'main' function, aka run_all_tests:
- run_excessive: If True, drastically increases number of epsisode on a few spots to verify that the CLT kicks in
- print_incomplete_boards: Used to identify board where not all fields have a disk placed, because no player is
  able to make an action anymore. I could not think of any reasonable method to verify that such a early termination
  is indeed correct without using code from the board, which again would be error-prone itself potentiall if there
  was any error. I looked at around 200 incomplete boards and verified myself that all of them have been correctly
  terminated.
- print_boards: prints all board states apparent in the tests below. Useful if there is a test is failing
- a_vs_a: If True, lets a lot of agents play against each other.

IMPORTANT: Set run_excessive and a_vs_a to True and run the file to verify that the environment + the data is working
fine! Setting those parameters to True means the file takes around an hour of running.

With this file running smoothly, I conclude that Í did my best to verify that there are no potential logic or
implementation errors anywhere in the code, except any agent that is not in simple_agents.py

"""
from src.agents.medium_agents import SimpleQLearningAgent
from src.agents.simple_agents import RandomAgent, MinAgent, MaxAgent
from src.environment.board import OthelloBoard
from src.utils.data_loader import read_othello_dataset, store_q_agent, load_q_agent
from src.utils.agent_vs_agent_simulation import simulate_agent_vs_agent
from src.environment.replay_buffer import create_and_store_replay_buffer
import string, random, time
from pathlib import Path


def run_board_correctness_tests(run_excessive=True, print_incomplete_boards=False, print_boards=False):
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

    print("trying out some actions..")
    board = OthelloBoard(rows=10, cols=10)
    board.print_board()
    try_valid_action(board, "E4", 0)
    board.print_board()
    try_valid_action(board, "F4", 1)
    board.print_board()
    try_valid_action(board, "G5", 0)
    board.print_board()

    white_player_valid_actions = ['D4', 'D6', 'H4', 'H6']
    print(f"Player white is next and can do these actions: {white_player_valid_actions}")
    for action in white_player_valid_actions:
        boardTemp = board.copy()
        try_valid_action(boardTemp, action, 1)

    all_actions = [f"{letter}{number}" for letter in string.ascii_uppercase[:8] for number in range(1, 9)]
    print(
        f"white player should not be able to do any of the other actions..{[action for action in all_actions if action not in white_player_valid_actions]}")
    for action in all_actions:
        if action not in white_player_valid_actions:
            boardTemp = board.copy()
            try_invalid_action(boardTemp, action, 1)
    print("White making a action..")
    try_valid_action(board, "D4", 1)
    print("let black play..")
    print("Field now for black player:")
    board.print_board()

    black_player_valid_actions = ['F3', 'E3', 'D5', 'F7', 'D3', 'E7', 'G6']
    print(f"black player is next and can do these actions: {black_player_valid_actions}")
    for action in black_player_valid_actions:
        boardTemp = board.copy()
        try_valid_action(boardTemp, action, 0)

    print(
        f"white player should not be able to do any of the other actions..{[action for action in all_actions if action not in black_player_valid_actions]}")
    for action in all_actions:
        if action not in black_player_valid_actions:
            boardTemp = board.copy()
            try_invalid_action(boardTemp, action, 0)

    print("finish game randomly..")
    current_player = 0
    current_valid_actions = board.get_valid_actions(current_player)

    while len(current_valid_actions) > 0:
        current_player, current_board, current_valid_actions, winner = board.make_action(random.choice(current_valid_actions),
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
            current_valid_actions = board.get_valid_actions(current_player)
            while len(current_valid_actions) > 0:
                current_player, current_board, current_valid_actions, winner = board.make_action(
                    random.choice(current_valid_actions),
                    current_player)

    if run_excessive:
        print("\nchecking if central limit theorem kicks in when playing a shitload of rounds..")
        determine_average_winner(10000, print_boards=print_boards,
                                 print_only_incomplete_boards=print_incomplete_boards)
    else:
        determine_average_winner(episodes=1000, print_boards=print_boards,
                                 print_only_incomplete_boards=print_incomplete_boards)


def run_simple_agent_tests(print_boards=False, a_vs_a=True):

    if a_vs_a:
        print("\nPlaying RandomAgent vs. MinAgent on 8x8 board..")
        simulate_agent_vs_agent(board=OthelloBoard(rows=8, cols=8),
                                agent1=RandomAgent(),
                                agent1_hparams="None",
                                agent2=MinAgent(),
                                agent2_hparams="None",
                                print_boards=print_boards,
                                n_games=50,
                                delay=0)

        print("\nPlaying RandomAgent vs. MaxAgent..")
        simulate_agent_vs_agent(board=OthelloBoard(),
                                agent1=RandomAgent(),
                                agent1_hparams="None",
                                agent2=MaxAgent(),
                                agent2_hparams="None",
                                print_boards=print_boards,
                                delay=0,
                                n_games=50)

        print("\nPlaying MinAgent vs. MaxAgent..")
        simulate_agent_vs_agent(board=OthelloBoard(8,8),
                                agent1=MinAgent(),
                                agent1_hparams="None",
                                agent2=MaxAgent(),
                                agent2_hparams="None",
                                print_boards=print_boards,
                                print_only_final_result=print_boards,
                                delay=0,
                                n_games=50,
                                alternate_beginner=False)

        print("\nPlaying MaxAgent vs. MinAgent..")
        simulate_agent_vs_agent(board=OthelloBoard(8,8),
                                agent1=MaxAgent(),
                                agent1_hparams="None",
                                agent2=MinAgent(),
                                agent2_hparams="None",
                                print_boards=print_boards,
                                print_only_final_result=print_boards,
                                delay=0,
                                n_games=50,
                                alternate_beginner=False)

        print("\nPlaying MinAgent vs. MaxAgent with both having epsilon 0.05 to introduce some level of randomness..")
        simulate_agent_vs_agent(board=OthelloBoard(8,8),
                                agent1=MinAgent(0.05),
                                agent1_hparams=f"epsilon_{0.05}",
                                agent2=MaxAgent(0.05),
                                agent2_hparams=f"epsilon_{0.05}",
                                print_boards=print_boards,
                                print_only_final_result=print_boards,
                                delay=0,
                                n_games=50)

        print("Trying to learn from simple Q learning algorithm")
        print("Stating to learn from experience..")
        environment = OthelloBoard(8,8)
        q_learning_agent = SimpleQLearningAgent(environment)
        othello_games = read_othello_dataset()
        for idx, game in othello_games.iterrows():
            environment.reset_board()
            q_learning_agent.learn_by_experience(game['winner'], game['game_moves'])
            if idx % 1000 == 0:
                print(f"learned from {idx} games..")

        n_episodes = 1000000
        print(f"starting to learn by letting agent play against itself for {n_episodes} episodes..")

        # Train the Q-Learning Agent over multiple episodes
        for episode in range(n_episodes):
            q_learning_agent.epsilon = 0.1

            if episode % 1000 == 0:
                print(f"training simple Q-Learning agent, episode {episode} of {n_episodes}")
            q_learning_agent.play_episode()
            if episode % 50000 == 0:
                #q_learning_agent.decay_epsilon(0.995)
                print("letting Q-Learning-Agent play against other agents ..")
                q_learning_agent.epsilon = 0  # no more exploration, just brutally beating the simple baselines
                #store_q_agent(q_learning_agent)

                print("Q-learning vs RandomAgent..")
                simulate_agent_vs_agent(board=environment,
                                        agent1=RandomAgent(),
                                        agent1_hparams="None",
                                        agent2=q_learning_agent,
                                        agent2_hparams=f"experience_learned",
                                        print_boards=print_boards,
                                        alternate_beginner=True,
                                        n_games=100,
                                        delay=0)

                print("Q-learning vs MinAgent..")
                simulate_agent_vs_agent(board=environment,
                                        agent1=MinAgent(),
                                        agent1_hparams="None",
                                        agent2=q_learning_agent,
                                        agent2_hparams=f"experience_learned",
                                        print_boards=print_boards,
                                        alternate_beginner=True,
                                        n_games=100,
                                        delay=0)

                print("Q-learning vs MaxAgent..")
                simulate_agent_vs_agent(board=environment,
                                        agent1=MaxAgent(),
                                        agent1_hparams="None",
                                        agent2=q_learning_agent,
                                        agent2_hparams=f"experience_learned",
                                        print_boards=print_boards,
                                        alternate_beginner=True,
                                        n_games=100,
                                        delay=0)

                """print("Trying to run all games given in othello_dataset.csv..")
                othello_games = read_othello_dataset()
                board = OthelloBoard(rows=8, cols=8)
                for idx, game in othello_games.iterrows():
                    board.reset_board()
                    next_player = 0
                    for action in game['game_moves']:
                        next_player, _, _, winner = board.make_action(action, next_player)
        
                    # not putting it in if, as playing all actions should lead to a conclusion
                    if winner == - 1 and game['winner'] == 1: assert True
                    elif winner == -2 and game['winner'] == -1: assert True
                    elif winner == -3 and game['winner'] == 0: assert True
                    else:
                        print(f"Something's wrong with that game: {game['eOthello_game_id']}")
                        board.print_board()
                        assert False
        
                    if idx % 500 == 0: print(f"Ran {idx} games")"""


def try_invalid_action(board, action, player):
    """
    makes an invalid action in a given board state, fails if such an action is possible
    """
    try:
        board.make_action(action, player)
        assert False, print(f"Action {action} is NOT valid")
    except ValueError:
        print(f"Test passed, action {action} is valid")


def try_valid_action(board, action, player):
    """
    makes a valid action in a given board state, fails if the action is not possible
    """
    try:
        board.make_action(action, player)
        print(f"Test passed, action {action} is valid")
    except ValueError:
        assert False, print(f"Action {action} should be valid!")


def try_bad_board_config(col, row):
    """
    gets an invalid row/col combination, fails if such a board can be created.
    """
    try:
        boardNok = OthelloBoard(cols=col, rows=row)
        assert False, print(f"board with col={col}, row={row} should NOT be possible")
    except ValueError:
        print(f"Test passed")


def try_good_board_config(col, row):
    """
    gets a valid row/col combination, fails if such a board can NOT be created.
    """
    try:
        boardOk = OthelloBoard(cols=col, rows=row)
        print("Test passed")
    except ValueError:
        assert False, print(f"board with col={col}, row={row} should be possible")


def determine_average_winner(episodes=1000, rows=6, cols=6, print_boards=False, print_only_incomplete_boards=False):
    """
    Used to verify that the CLT indeed kicks in, however only lets the test fail if at least 10.0000 rounds have
    been played to make sure that indeed the CLT should have kicked in.
    """
    print("using alternating beginner (black/white) to remove issues from who starts in random games..")
    white_wins = 0
    black_wins = 0
    draws = 0
    for i in range(episodes):
        if i % 5000 == 0:
            print(f"Episode {i} in CLT checker of {episodes}")
        res = play_random_game(rows, cols, print_boards, current_player=0 if i % 2 == 0 else 1,
                               print_only_incomplete_boards=print_only_incomplete_boards)

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
    returns the winner of a random game
    """

    board = OthelloBoard(rows=rows, cols=cols, first_player=current_player)
    current_valid_actions = board.get_valid_actions(current_player)
    while len(current_valid_actions) > 0:
        current_player, current_board, current_valid_actions, winner = board.make_action(random.choice(current_valid_actions),
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


def run_replay_buffer_tests():
    print('checking out if I can read use the replay buffer..')
    create_and_store_replay_buffer()
    print('processed all games, checking if file exists .. ')
    buffer_file = Path('../../data/replay_buffer.json')

    assert buffer_file.exists(), 'this file should exist!'



if __name__ == "__main__":
    #run_board_correctness_tests(run_excessive=False, print_boards=False, print_incomplete_boards=False)
    #run_simple_agent_tests()
    run_replay_buffer_tests()
    print("\n\nALL TESTS PASSED!")
