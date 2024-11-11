import time, inspect
from src.utils.results_writer import store_results


def simulate_agent_vs_agent(board,
                            agent1,
                            agent1_hparams,
                            agent2,
                            agent2_hparams,
                            print_boards,
                            n_games,
                            delay=0.1,
                            print_only_final_result=False,
                            alternate_beginner=True):
    """
    This function is used for simulating two agents playing against each other.
    Results are stored

    :param board: OthelloBoard
    :param agent1: black agent
    :param agent1_hparams: further description of black agent (hparams)
    :param agent2: white agent
    :param agent2_hparams: further description of white agent (hparams)
    :param print_boards: If True, prints all boards
    :param print_only_final_result: If True, prints final result
    :param n_games: how many games the agents will play against each other
    :param delay: delay in second if something is printed
    :param alternate_beginner: if False, black always begins. Interesting to check
                                potential advantages regarding the beginner

    :return: None
    """
    sig = inspect.signature(simulate_agent_vs_agent)

    # Check if all parameters have been passed
    try:
        bound_args = sig.bind(board, agent1, agent1_hparams, agent2, agent2_hparams,
                              print_boards, n_games, delay, print_only_final_result)
    except TypeError as e:
        raise TypeError("Missing parameters:", e)
    print(f"Starting to play {agent1.__class__.__name__} vs. {agent2.__class__.__name__} with {n_games} episodes.") if print_boards else None
    print(f"Params for {agent1.__class__.__name__}: {agent1_hparams}") if print_boards else None
    print(f"Params for {agent2.__class__.__name__}: {agent2_hparams}") if print_boards else None

    agent1_wins, agent2_wins, draws = 0, 0, 0

    for episode in range(n_games):
        print(f"playing Episode {episode} of {n_games}..") if print_boards else None
        board.reset_board()

        # Case 1: Agent1 always begins
        if not alternate_beginner:
            current_black_agent = agent1
            current_white_agent = agent2

        # Case 2: Alternating beginner, decide by modulo who starts this round
        else:
            if episode % 2 == 0:
                current_black_agent = agent1
                current_white_agent = agent2
            else:
                current_white_agent = agent1
                current_black_agent = agent2

        print(f"{current_black_agent.__class__.__name__} begins with first move") if print_boards else None

        winner = simulate_agent_vs_agent_single_episode(board=board,
                                                        black=current_black_agent,
                                                        white=current_white_agent,
                                                        print_boards=print_boards,
                                                        print_only_final_result=print_only_final_result,
                                                        episode=episode,
                                                        delay=delay)

        if winner == -1:  # black won
            if not alternate_beginner:
                agent1_wins += 1
            else:
                if episode % 2 == 0:
                    agent1_wins += 1
                else:
                    agent2_wins += 1
        elif winner == -2:  # white won
            if not alternate_beginner:
                agent2_wins += 1
            else:
                if episode % 2 == 0:
                    agent2_wins += 1
                else:
                    agent1_wins += 1
        else:  # Draw
            draws += 1

        if print_boards and print_only_final_result:
            print(f"winner is {winner}") if print_boards else None
            board.print_board()
            time.sleep(delay)

    # Print results after all episodes
    print(f"Results after {n_games} episodes: {agent1.__class__.__name__} wins: {agent1_wins}, {agent2.__class__.__name__} wins: {agent2_wins}, Ties: {draws}") if print_boards else None

    # Store overall in out-file
    store_results(agent1_wins=agent1_wins,
                  agent2_wins=agent2_wins,
                  draws=draws,
                  n_games=n_games,
                  agent1_name=agent1.__class__.__name__,
                  agent2_name=agent2.__class__.__name__,
                  alternating_starts=alternate_beginner,
                  agent1_hparams=agent1_hparams,
                  agent2_hparams=agent2_hparams)


def simulate_agent_vs_agent_single_episode(board,
                                           black,
                                           white,
                                           print_boards,
                                           print_only_final_result,
                                           episode,
                                           delay):
    current_player, winner = 0, 0

    while True:

        # Choose the agent based on the current player
        agent = black if current_player == 0 else white

        action = agent.get_action(board, current_player)

        # Make the move
        next_player, _, _, winner = board.make_action(action, current_player)

        # Print board if requested
        if print_boards and not print_only_final_result:
            print(f"Episode {episode + 1}, {['Black', 'White'][current_player]}'s move:") if print_boards else None
            board.print_board()
            time.sleep(delay)

        # If the game is over, print and return the winner
        if winner < 0:
            if winner == -1:
                print("Black wins this game!") if print_boards else None
            elif winner == -2:
                print("White wins this gme!") if print_boards else None
            else:
                print("It's a tie!") if print_boards else None
            return winner

        current_player = next_player
