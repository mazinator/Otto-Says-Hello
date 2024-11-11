"""
used when agent vs. agent is played. Results + metadata stored continuously in the file for a later analysis
"""

import csv, os

OUT_FILE = "../../out/agent_vs_agent_results.csv"


def store_results(black_wins, white_wins, draws, episodes, black_agent, white_agent, out_file=OUT_FILE):
    """
    Stores the game results continuously to a CSV file.

    Args:
        black_wins (int): Number of wins by the black player.
        white_wins (int): Number of wins by the white player.
        draws (int): Number of draws.
        episodes (int): Number of games played.
        black_agent (str): Name of the black player agent.
        white_agent (str): Name of the white player agent.
        out_file (str): Path to the output CSV file.
    """
    file_exists = os.path.isfile(out_file)

    headers = ["Black Agent", "White Agent", "Episodes", "Black Wins", "White Wins", "Draws"]
    row = [black_agent, white_agent, episodes, black_wins, white_wins, draws]

    with open(out_file, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        writer.writerow(row)