"""
Script to play against an agent of your choice.

IMPORTANT: do not change the field size! Leave it at 8x8, otherwise there will be errors with the neural net.
"""


import argparse
from src.agents.simple_agents import *
from src.agents.alpha_zero import *
from src.environment.board import *
from src.utils.nn_helpers import get_device

# Note: CUT STARTING AT episodes!
BEST_MODEL = 'cp_alphazero_final_0.001_lr'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opponent',
                        default='alphazero',
                        choices=['maxMove', 'minMove', 'randomMove', 'alphazero'],
                        help='Selection of opposing agent')
    parser.add_argument('--colorOwn',
                        default='black',
                        choices=['black', 'white'],
                        help='Selection of own color')
    parser.add_argument('--startColor',
                        default='black',
                        choices=['black', 'white'],
                        help='Which color starts (normally always black)')
    parser.add_argument('--fieldSize',
                        default=8,
                        help='An even number between 6 and 26 indicating the field size')
    return parser.parse_args()


def get_agent(args):
    agent_arg = args.opponent
    if agent_arg == 'maxMove':
        return MaxAgent()
    elif agent_arg == 'minMove':
        return MinAgent()
    elif agent_arg == 'randomMove':
        return RandomAgent()
    elif agent_arg == 'alphazero':
        device = get_device()
        model = AlphaZeroNetWithResiduals(args.fieldSize, args.fieldSize).to(device)
        checkpoint_path, episode_loaded = get_checkpoint(BEST_MODEL, '.checkpoints')
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded AlphaZero model from episode {checkpoint['episode']} (prefix was {BEST_MODEL})")
        else:
            raise ValueError('error loading a trained model!')
        return model


def play_game(args):
    agent = get_agent(args)
    color_human = 0 if args.colorOwn == 'black' else 1  # Human color as integer
    color_agent = 1 - color_human  # Opponent color
    first_player = 0 if args.startColor == 'black' else 1
    board = OthelloBoard(rows=args.fieldSize, cols=args.fieldSize, first_player=first_player)

    current_player = first_player

    print(f"Let's start the game! You play against: {agent.__class__.__name__}")

    while True:
        print(f"\nIt's {['black', 'white'][current_player]}'s move:")
        board.print_board()

        valid_actions = board.get_valid_actions(current_player)
        if not valid_actions:
            print(f"No valid moves for {['black', 'white'][current_player]}. Skipping turn.")
            current_player = 1 - current_player
            if not board.get_valid_actions(current_player):  # Check if both players cannot move
                print("No valid moves for either player. Game over.")
                break
            continue

        if current_player == color_human:
            print("Your available moves:", [f"{chr(c + ord('A'))}{r + 1}" for r, c in valid_actions])
            while True:
                try:
                    move = input("Enter your move (e.g., A1): ").strip()
                    if len(move) < 2:
                        raise ValueError("Invalid input format. Use 'A1' or similar.")
                    col = ord(move[0].upper()) - ord('A')
                    row = int(move[1:]) - 1
                    if (row, col) not in valid_actions:
                        raise ValueError("Invalid move. Try again.")
                    board.make_action((row, col), current_player)
                    break
                except Exception as e:
                    print(e)
        else:
            print(f"{agent.__class__.__name__} ({['black', 'white'][current_player]}) is thinking...")
            move = agent.get_action(board=board, player=current_player)
            print(f"Agent chose: {chr(move[1] + ord('A'))}{move[0] + 1}")
            board.make_action(move, current_player)

        # Determine if the next player has valid moves
        next_player = 1 - current_player
        if not board.get_valid_actions(next_player):
            print(
                f"{['black', 'white'][next_player]} has no valid moves. {['black', 'white'][current_player]} plays again.")
        else:
            current_player = next_player  # Switch turns if the next player has valid moves

    board.print_board()
    winner = board.check_winner()
    stone_counts = board.stone_count()
    if winner == -1:
        print(f"Black wins {stone_counts['black']}:{stone_counts['white']}!")
        print("Congrats!") if color_human == 'black' else None
    elif winner == -2:
        print(f"White wins {stone_counts['white']}:{stone_counts['black']}!")
        print("Congrats!") if color_human == 'white' else None
    else:
        print("It's a tie!")

    new_round = input("New round? (Y/y/Yes or anything else for no)").strip()

    if new_round in ['y', 'Y', 'Yes']:
        play_game(args)


def main():
    args = parse_args()
    play_game(args)


if __name__ == '__main__':
    main()
