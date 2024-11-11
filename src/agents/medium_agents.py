"""
This file contains some more sophisticated agents, i.e. SARSA and Q-Learning without a Deep neural approach

- Q-learning will learn on the human-played games (possible as it is off-policy)
- SARSA is an on-policy algorithm and can therefore not learn from experience
"""
import random


class SimpleQLearningAgent:
    def __init__(self, board, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.board = board
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self):
        """
        Convert the current board state to a unique tuple representation for Q-table indexing.
        """
        return tuple(self.board.board.flatten())

    def get_action(self, board, player):
        """
        Choose an action based on epsilon-greedy policy.

        Args:
            player (int): The current player (0 for black, 1 for white).

        Returns:
            tuple: Chosen action as (row, col).
        """
        state = self.get_state()
        valid_moves = self.board.get_valid_actions(player)

        # If there are no valid moves, return None
        if not valid_moves:
            raise ValueError(f'No valid moves found for player {player}')

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice(valid_moves)
        else:  # Exploitation
            q_values = [self.q_table.get((state, move), 0) for move in valid_moves]
            max_q_value = max(q_values)
            best_moves = [move for move, q in zip(valid_moves, q_values) if q == max_q_value]
            return random.choice(best_moves)  # Choose randomly among moves in case of ties

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair using the Q-learning formula.

        Args:
            state (tuple): Current state.
            action (tuple): Action taken.
            reward (float): Reward received after taking the action.
            next_state (tuple): Next state after taking the action.
        """
        current_q = self.q_table.get((state, action), 0)
        future_rewards = [self.q_table.get((next_state, a), 0) for a in self.board.get_valid_actions(self.board.next_player)]

        # Q-learning formula
        max_future_q = max(future_rewards) if future_rewards else 0
        td_target = reward + self.discount_factor * max_future_q
        td_error = td_target - current_q
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

    def decay_epsilon(self, decay_rate):
        """
        Decay the epsilon value over time to reduce exploration as the agent learns.

        Args:
            decay_rate (float): The rate at which to decay epsilon.
        """
        self.epsilon *= decay_rate

    def play_episode(self):
        """
        Play one episode (one complete game) with the current Q-learning policy.
        """
        self.board.reset_board()
        current_player = self.board.next_player
        state = self.get_state()
        winner = 0  # No winner at the beginning

        while winner == 0:
            action = self.get_action(None, current_player)

            if action is None:  # No valid moves, switch to the other player
                current_player = 1 if current_player == 0 else 0
                continue

            # Execute the action and observe the reward and new state
            next_player, _, _, winner = self.board.make_action(action, current_player)
            next_state = self.get_state()

            # Define reward: +1 for winning, -1 for losing, 0 otherwise
            if winner == -1 and current_player == 0:  # Black wins
                reward = 1
            elif winner == -2 and current_player == 1:  # White wins
                reward = 1
            elif winner == -3:  # Draw, at least indicate that a draw is better than a loss
                reward = 0.5
            else:
                reward = 0

            # Update Q-value
            self.update_q_value(state, action, reward, next_state)

            # Move to the next state and player
            state = next_state
            current_player = next_player
