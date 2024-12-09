from sympy.stats.rv import probability

from src.agents.alpha_zero import AlphaZeroNet
from src.environment.board import OthelloBoard
from src.utils.nn_helpers import *
import math
from multiprocessing import Pool
import os


class MCTS:
    def __init__(self, model, simulations=500, exploration_rate=1000):
        """
        This class performs Monte Carlo Tree Search based on the Upper Confidence Bound (UCB).

        :param board: OthelloBoard object
        :param model: AlphaZeroNet object
        :param simulations: int
        :param exploration_rate: per default sqrt(2)
        """
        self.model = model  # AlphaZeroNet model
        self.simulations = simulations  # int

        # this dictionaries contains infos about states and state-actions pairs, two types of indexes used!
        self.visits = {}  # indicates if this state was already visited

        # the 2 dictionaries are all hashed dictionaries with the board states as keys
        self.priors = {}  # probability distribution over all 64 fields in a given state
        self.children = {}  # actions possible in a given state as list of tuples (row, col)

        # This dictionary is indexed by (state, action) and (state), with the value being the average reward
        self.values = {}

        self.exploration_rate = exploration_rate  # Hyperparameter for the UCB

        self.cached_predictions = {}

    def search(self, current_board, player, device='çpu'):
        """
        TODO function not used, only self reference!
        :param current_board:
        :param player:
        :param device:
        :return:
        """

        # Check if some other child-node already reached that state
        if current_board not in self.visits:

            # Get all child nodes of current node
            valid_actions = current_board.get_valid_actions(player)

            # Backpropagate result if terminal state is reached
            if valid_actions is None:
                return -self.evaluate(current_board, player, device)

            # Store valid actions for a given state
            self.children[current_board] = valid_actions

            board_tensor = prepare_alphazero_board_tensor(current_board.board, player).to(device)

            # Use model to get probability distribution for n
            with torch.no_grad():
                p, v = self.model(board_tensor)
            self.priors[current_board] = p.cpu().squeeze().detach().numpy()
            self.values[current_board] = 0
            self.visits[current_board] = 0
            return -v.item()

        best_ucb = -float("inf")
        best_action = None

        # Use Upper Confidence Bound for Trees (UCT)
        sqrt_visits = math.sqrt(self.visits[current_board])
        for action in self.children[current_board]:
            q = self.values.get((current_board, action), 0)
            n = self.visits.get((current_board, action), 0)
            prior = self.priors[current_board][action[0] * action[1]]
            ucb = q + self.exploration_rate * prior * sqrt_visits / (1 + n)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        # Make the action
        if best_action is not None:
            new_board = current_board.copy()
            new_player, _, _, _ = new_board.make_action(best_action, player)
            result = -self.search(new_board, new_player, device)

            # Update visit count and value
            self.visits[(current_board, best_action)] = self.visits.get((current_board, best_action), 0) + 1
            self.visits[current_board] += 1
            self.values[(current_board, best_action)] = (
                    (self.values.get((current_board, best_action), 0) + result) / self.visits[(current_board, best_action)]
            )

            return result
        else:
            # No valid moves, return evaluation
            return -self.evaluate(current_board, player, device)

    def evaluate(self, current_board, player, device):
        state_key = (current_board, player)
        if state_key in self.cached_predictions:
            return self.cached_predictions[state_key]

        board_tensor = prepare_alphazero_board_tensor(current_board.board, player).to(device)
        _, v = self.model(board_tensor)
        self.cached_predictions[state_key] = v.item()
        return v.item()

    def get_policy(self, board, tau=1):
        """
        Get the policy distribution over all 64 possible actions as an 8x8 grid.

        :param board: Current board state.
        :param tau: Temperature parameter to adjust exploration (tau=1 for softmax, tau=0 for greedy).
        :return: An 8x8 probability grid for all possible moves.
        """
        # Initialize a zero grid for all 64 possible actions in an 8x8 format
        policy = np.zeros((8, 8))

        # Get valid moves and their visit counts
        valid_moves = self.children.get(board, [])  # List of valid actions
        visit_counts = np.array([self.visits.get((board, move), 0) for move in valid_moves])

        if len(valid_moves) == 0:
            # No valid moves available, return zero-filled grid
            return policy

        # Apply temperature to the visit counts
        if tau == 0:
            # Greedy: Choose the move with the highest visit count
            best_move = valid_moves[np.argmax(visit_counts)]
            policy[best_move[0], best_move[1]] = 1.0
        else:
            # Apply softmax temperature scaling
            visit_counts = visit_counts ** (1 / tau)
            total_count = sum(visit_counts)
            probabilities = visit_counts / visit_counts.sum()

            # Assign probabilities to valid actions in the policy grid
            for move, prob in zip(valid_moves, probabilities):
                policy[move[0], move[1]] = prob

        return policy

    def run_simulations(self, board, player, device):
        """
        Runs the specified number of simulations from the root node using parallel processing.

        :param board: The root state (node) of the game.
        :param player: The current player.
        :param device: The device
        """
        for i in range(self.simulations):
            self.search(current_board=board, player=player, device=device)
