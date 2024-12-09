from __future__ import division
from mcts.searcher.mcts import MCTS

from src.agents.alpha_zero import AlphaZeroNet
from src.environment.board import OthelloBoard
import torch

from src.utils.nn_helpers import prepare_alphazero_board_tensor, get_device


class OthelloMCTS:
    def __init__(self, board, player, model=None):
        self.board = board.copy()
        self.current_player = player
        self.model = model  # AlphaZero neural network

    def get_current_player(self):
        return self.current_player

    def get_possible_actions(self):
        """
        Returns a list of all valid actions for the current player.
        Each action is represented as a tuple (row, col).
        """
        return self.board.get_valid_actions(self.current_player)

    def take_action(self, action):
        """
        Simulates taking an action on the board and returns the resulting state.
        :param action: The action to take, represented as a tuple (row, col).
        :return: A new OthelloMCTS instance representing the resulting state.
        """
        new_board = self.board.copy()
        player, _, _, _ = new_board.make_action(action, self.current_player)
        return OthelloMCTS(board=new_board, player=player, model=self.model)

    def is_terminal(self):
        """
        Checks if the game is in a terminal state (no valid actions for both players).
        :return: True if the game is over, False otherwise.
        """
        current_valid = self.board.get_valid_actions(self.current_player)
        other_valid = self.board.get_valid_actions(1 if self.current_player == 0 else 0)
        return len(current_valid) == 0 and len(other_valid) == 0

    def get_reward(self):
        """
        Calculates the reward for the current player. Rewards are based on disk count:
        - Positive reward if the current player wins.
        - Negative reward if the opponent wins.
        - Zero reward for a draw.
        :return: Reward as a float.
        """
        winner = self.board.check_winner()
        if winner == -1:  # Black wins
            return 1.0 if self.current_player == 0 else -1.0
        elif winner == -2:  # White wins
            return 1.0 if self.current_player == 1 else -1.0
        else:  # Draw
            return 0.0

    def get_policy_distribution(self):
        """
        Uses the AlphaZero model to compute the policy distribution over all actions.
        :return: A probability distribution over all possible actions.
        """
        if not self.model:
            raise ValueError("AlphaZero model is not provided.")

        # Prepare the board tensor for the model
        board_tensor = prepare_alphazero_board_tensor(self.board.board, self.current_player)

        # Get policy and value from the model
        with torch.no_grad():
            policy, _ = self.model(board_tensor)

        # Convert policy logits to probabilities
        policy_probs = torch.exp(policy).cpu().numpy().flatten()

        # Get valid actions and mask invalid ones
        valid_moves = self.get_possible_actions()
        valid_mask = [0] * (self.board.rows * self.board.cols)

        for move in valid_moves:
            index = move[0] * self.board.cols + move[1]
            valid_mask[index] = 1

        # Apply mask and normalize
        masked_probs = policy_probs * valid_mask
        total_prob = sum(masked_probs)
        if total_prob > 0:
            masked_probs /= total_prob

        return masked_probs

class Action():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))

if __name__ == "__main__":
    # Initialize the AlphaZero model
    device = get_device()
    model = AlphaZeroNet(8, 8).to(device)

    # Initialize the board and wrap it with MCTS
    initial_state = OthelloMCTS(OthelloBoard(8, 8), model=model)

    # Initialize MCTS
    searcher = MCTS(time_limit=1000)

    # Get the policy distribution from the AlphaZero model
    policy_distribution = initial_state.get_policy_distribution()

    print("Policy Distribution:", policy_distribution)