"""
used when agent vs. agent is played. Results + metadata stored continuously in the file for a later analysis
"""

import csv, os, time

OUT_FILE = "../../out/agent_vs_agent_results.csv"


def store_results(agent1_wins,
                  agent2_wins,
                  draws,
                  n_games,
                  agent1_name,
                  agent1_hparams,
                  agent2_name,
                  agent2_hparams,
                  alternating_starts,
                  out_file=OUT_FILE):
    """
    Stores the game results continuously to a CSV file.

    :param agent1_wins: Number of wins by the black player.
    :param agent2_wins: Number of wins by the white player.
    :param draws: Number of draws.
    :param n_games: Number of games played.
    :param agent1_name: Name of the black player agent.
    :param agent2_name: Name of the white player agent.
    :param out_file: Path to the output CSV file.
    :param agent1_hparams: hparams of black
    :param agent2_hparams: hparams of white
    :param alternating_starts: boolean if the two agents had alternating starts
    """
    file_exists = os.path.isfile(out_file)

    headers = ["Black Agent", "White Agent", "Episodes", "Black Wins", "White Wins", "Draws",
               "Agent1_description", "Agent2_description", "Alternating Starts"]
    row = [agent1_name, agent2_name, n_games, agent1_wins, agent2_wins, draws,
           agent1_hparams, agent2_hparams, alternating_starts, time.time()]

    with open(out_file, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(row)
