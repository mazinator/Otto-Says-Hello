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

OTHELLO_DATASET = "../../data/othello_dataset.csv"


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
