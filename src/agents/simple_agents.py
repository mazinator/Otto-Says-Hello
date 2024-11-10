import random


class RandomAgent:

    def get_action(self, board, player):
        return random.choice(board.get_valid_moves(player=player))


class MinAgent:
    def __init__(self, epsilon=0):
        self.epsilon = epsilon

    def get_action(self, board, player):
        """
        the min_agent always chooses the action which leads to the least stones flipped
        """

        valid_moves = board.get_valid_moves(player)

        if not valid_moves:
            return None  # No valid moves available

        # With epsilon probability, pick a random move for exploration
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Find the move with the minimum flips
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
        the min_agent always chooses the action which leads to the least stones flipped
        """

        valid_moves = board.get_valid_moves(player)

        if not valid_moves:
            return None  # No valid moves available

        # With epsilon probability, pick a random move for exploration
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Find the move with the minimum flips
        max_flips = float('-inf')
        best_move = None

        for move in valid_moves:
            flips = board.simulate_flip_count(move, player)
            if flips > max_flips:
                max_flips = flips
                best_move = move

        return best_move

