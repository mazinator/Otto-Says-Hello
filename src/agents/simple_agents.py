"""
This file implements some very stupid agents, they were mostly used for testing purposes and to have
some feedback for the medium agents, like simple Q-Learning without any neural network addition.
"""

import random


class RandomAgent:

    def get_action(self, board, player):
        return random.choice(board.get_valid_actions(player=player))


class MinAgent:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon

    def get_action(self, board, player):
        """
        the min_agent always chooses the action which leads to the least stones flipped
        """

        valid_moves = board.get_valid_actions(player)

        if not valid_moves:
            assert False, "this function should not have been called if this agent has no moves"

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        min_flips = float('inf')
        best_move = None

        for move in valid_moves:
            flips = board.simulate_flip_count(move, player)
            if flips < min_flips:
                min_flips = flips
                best_move = move

        return best_move


class MaxAgent:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon

    def get_action(self, board, player):
        """
        the max_agent always chooses the action which leads to the least stones flipped
        """

        valid_moves = board.get_valid_actions(player)

        if not valid_moves:
            assert False, "this function should not have been called if this agent has no moves"

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        max_flips = float('-inf')
        best_move = None

        for move in valid_moves:
            flips = board.simulate_flip_count(move, player)
            if flips > max_flips:
                max_flips = flips
                best_move = move

        return best_move

