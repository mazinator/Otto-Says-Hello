"""
IMPORTANT!
I later that trying to solve such a problem with a simple Q-learning agent
was hopeless from the beginning, the search space is just to big especially if you train
on a laptop. I left this as a proof for 'I tried'.


This file contains some more sophisticated agents, i.e. SARSA and Q-Learning without a Deep neural approach

- Q-learning will learn on the human-played games (possible as it is off-policy)
- SARSA is an on-policy algorithm and can therefore not learn from experience

NOTE: Due to the fact that I already invested like 40 hours and didn't
even start yet with the neural network, I reference to the realm of papers that exist which
indicate that a simple SARSA algorithm would be as bad as simple Q-learning, besides the fact
that SARSA cannot even make use of the ReplayBuffer as it is an on-policy algorithm.
"""
import random


class SimpleQLearningAgent:
    def __init__(self, board, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.board = board
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self, player):
        """
        Convert the current board state to a unique tuple representation for Q-table indexing,
        including player information.
        """
        return tuple(self.board.board.flatten())

    def get_action(self, board, player):
        """
        Choose an action based on epsilon-greedy policy.

        :param board:
        :param player: The current player (0 for black, 1 for white).

        :returns: tuple: Chosen action as (row, col).
        """
        state = self.get_state(player)
        valid_moves = self.board.get_valid_actions(player)

        if not valid_moves:
            return None

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice(valid_moves)
        else:  # Exploitation
            q_values = [self.q_table.get((state, move), 0) for move in valid_moves]
            max_q_value = max(q_values)
            best_moves = [move for move, q in zip(valid_moves, q_values) if q == max_q_value]
            return random.choice(best_moves)

    def update_q_value(self, player, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair using the Q-learning formula.

        :param player (int): The player who took the action.
        :param state (tuple): Current state.
        :param action (tuple): Action taken.
        :param reward (float): Reward received after taking the action.
        :param next_state (tuple): Next state after taking the action.
        """
        current_q = self.q_table.get((state, action), 0)
        future_rewards = [self.q_table.get((next_state, a), 0) for a in self.board.get_valid_actions(self.board.next_player)]

        max_future_q = max(future_rewards) if future_rewards else 0
        td_target = reward + self.discount_factor * max_future_q
        td_error = td_target - current_q
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

    def decay_epsilon(self, decay_rate):
        self.epsilon *= decay_rate

    def play_episode(self):
        """
        Play one episode (one complete game) with the current Q-learning policy.
        """
        self.board.reset_board()
        current_player = self.board.next_player
        state = self.get_state(current_player)
        winner = 0

        while winner == 0:
            action = self.get_action(None, current_player)

            if action is None:
                current_player = 1 if current_player == 0 else 0
                continue

            next_player, _, _, winner = self.board.make_action(action, current_player)
            next_state = self.get_state(next_player)

            if winner == -1 and current_player == 0:
                reward = 1
            elif winner == -2 and current_player == 1:
                reward = 1
            elif winner ==  -3:
                reward = 0.5
            else:
                reward = 0

            self.update_q_value(current_player, state, action, reward, next_state)
            state = next_state
            current_player = next_player

    def learn_by_experience(self, result, moves):
        """
        Learn from a sequence of moves and game result to update the Q-table.

        :param result (int): The game outcome, 1 if black won, -1 if white won, 0 if a draw.
        :param moves (list): A list of moves as strings (e.g., ['F5', 'D6', ...]).
                          Black moves first, alternating turns.
        """
        reward_black = 1 if result == 1 else (0.5 if result == 0 else -1)
        reward_white = 1 if result == -1 else (0.5 if result == 0 else -1)

        # Initializing the board to replay moves
        self.board.reset_board()
        current_player = 0  # Black starts

        for i, action in enumerate(moves):
            # Convert the move from notation (e.g., 'F5') to board coordinates
            #action = self.board.convert_move_to_coordinates(move)  # Assuming this method exists

            # Get the current state and next state
            state = self.get_state(current_player)
            next_player, _, _, _ = self.board.make_action(action, current_player)
            next_state = self.get_state(next_player)

            # Assign reward only on the final move, using the result for both players
            reward = reward_black if current_player == 0 else reward_white if i == len(moves) - 1 else 0

            # Update Q-value for the current state-action pair
            self.update_q_value(current_player, state, action, reward, next_state)

            # Move to the next state and player
            current_player = next_player


