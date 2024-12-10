"""
I was able to gather around 25.000 played games from kaggle:
https://www.kaggle.com/datasets/andrefpoliveira/othello-games
There are also a few other data sources not used (see not_used.zip),
as there sometimes white started (definitely not a mistake of the
indication who black/white is, as the starting move could only have
been made by white given that one has to flip at least one stone.
I have decided to not include them in the experience replay, as I
already have around 25.000 qualitative games which I think should be
enough. Having a few more rounds makes the Kraut not fatter.
"""

import pandas as pd
import json

from src.agents.medium_agents import SimpleQLearningAgent


OTHELLO_DATASET = "../../data/othello_dataset.csv"
PATH_SIMPLE_Q_LEARNER = "../../models/simple_q_learner.json"


def read_othello_dataset(path=OTHELLO_DATASET):
    othello_data = pd.read_csv(path)
    othello_data['game_moves'] = othello_data['game_moves'].apply(extract_moves_from_stream)

    return othello_data


def extract_moves_from_stream(moves_stream: str):
    """
    Converts the stream of moves (e.g. "e3f4....") to a list of moves.
    Note: it would be computationally less expensive to directly play
    from that stream. However, this function was originally implemented
    to let the games be played in the test.py, to verify that none of the
    games is broken (as I already knew at that point that test.py and
    the board itself work fine). Don't change a running system.
    """
    moves = []

    i = 0
    while i < len(moves_stream):
        move = moves_stream[i:i + 2]

        if len(move) == 2 and move[0].isalpha() and move[1].isdigit():
            move = move[0].upper() + move[1]
            moves.append(move)
            i += 2
        else:
            raise ValueError(f"Invalid format found at row {i}: {move}")

    return moves


def store_q_agent(q_agent, filename=PATH_SIMPLE_Q_LEARNER):
    """
    Store the Q-learning agent's Q-table and parameters to a file.

    Args:
        q_agent (SimpleQLearningAgent): The Q-learning agent instance to store.
        filename (str): The filename where data will be stored.
    """
    # Prepare the data to be saved
    data = {
        "learning_rate": q_agent.learning_rate,
        "discount_factor": q_agent.discount_factor,
        "epsilon": q_agent.epsilon,
        "q_table": {str(k): v for k, v in q_agent.q_table.items()}
    }

    # Write the data to a JSON file
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Agent data stored in {filename}")


def load_q_agent(board, filename=PATH_SIMPLE_Q_LEARNER):
    """
    Load the Q-learning agent's Q-table and parameters from a file.

    Args:
        board: The board object the agent will use (needs to be provided on load).
        filename (str): The filename from which to load the agent data.

    Returns:
        SimpleQLearningAgent: A new instance of SimpleQLearningAgent with loaded parameters and Q-table.
    """
    # Load the data from the JSON file
    with open(filename, "r") as file:
        data = json.load(file)

    # Create a new agent with the loaded parameters
    q_agent = SimpleQLearningAgent(
        board=board,
        learning_rate=data["learning_rate"],
        discount_factor=data["discount_factor"],
        epsilon=data["epsilon"]
    )

    # Reconstruct the Q-table, converting keys back to tuples
    q_agent.q_table = {eval(k): v for k, v in data["q_table"].items()}

    print(f"Agent data loaded from {filename}")
    return q_agent

