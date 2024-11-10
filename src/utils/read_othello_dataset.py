import pandas as pd
OTHELLO_DATASET = "../../data/human_played_games/othello_dataset.csv"

def read_othello_dataset(path=OTHELLO_DATASET):
    othello_data = pd.read_csv(path)
    othello_data['game_moves'] = othello_data['game_moves'].apply(extract_moves_from_stream)

    return othello_data


def extract_moves_from_stream(moves_stream: str):
    moves = []

    # Assuming each move is two characters long, like "A1", "B2"
    i = 0
    while i < len(moves_stream):
        move = moves_stream[i:i + 2]

        # Ensure we have a valid move format (e.g., a letter followed by a digit)
        if len(move) == 2 and move[0].isalpha() and move[1].isdigit():
            move = move[0].upper() + move[1]
            moves.append(move)
            i += 2  # Move to the next pair
        else:
            print(f"Invalid move format found at position {i}: {move}")
            i += 1  # Skip to the next character in case of an error

    return moves
