"""
This file contains the code to create, store and load the replay buffer.

The replay buffer is a tuple consisting of:
- state s (8x8 field, so 64 bytes)
- action a (2 integers, so 2 byte)
- reward for performing a in s (mostly zero expect when it's the last move, 1 byte)
- next state s' (again 64 bytes)
- done flag (1 byte)

This leads to entries of around 132 bytes.
I have around 25.000 full games from eOthello, with mostly 60 moves
I do not use a maxCapacity for my ReplayBuffer, as all games lead to
a file size of 850MB, which is still reasonable for my laptop.
"""

import random, torch
from collections import deque
import numpy as np
import json, os

from mcts.searcher.mcts import MCTS

from src.agents.alpha_zero import AlphaZeroNetWithResiduals
from src.environment.board import OthelloBoard
from src.utils.data_loader import read_othello_dataset
from pathlib import Path

from src.utils.mcts_wrapper import OthelloMCTS
from src.utils.nn_helpers import get_device

OUT_FILE = 'data/replay_buffer.json'


class ReplayBuffer:
    def __init__(self, capacity=40000):
        """
        Initialize the replay buffer.

        :param capacity: Maximum number of experiences the buffer can store.
        2 Mio's capacity as default allow for enough room to keep all experiences
        from the 25.000 games which I have extracted from online
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, game_over):
        """
        Add a new experience to the buffer.

        :param state: Current state (board configuration).
        :param action: Action taken by the player (row, col).
        :param reward: Reward received for the action.
        :param next_state: Board configuration after the action.
        :param game_over: Whether the game is over.
        """
        self.buffer.append((state, action, reward, next_state, game_over))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of experiences to sample.

        :returns: tuple of form: (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        """
        Get the current size of the buffer.
        """
        return len(self.buffer)


class ReplayBufferAlphaZero:
    def __init__(self, capacity=40000):
        """
        Initialize the replay buffer.

        :param capacity: Maximum number of experiences the buffer can store.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, target_policy, reward):
        """
        Add a new experience to the buffer.

        :param state: Current state (board configuration).
        :param target_policy: TODO
        :param reward: Reward received for the action.
        """
        self.buffer.append((state, target_policy, reward))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of experiences to sample.

        :returns: tuple of form: (states, target_policies, rewards, next_states, dones).
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        """
        Get the current size of the buffer.
        """
        return len(self.buffer)

    def save_buffer(self, folder_path, filename):
        """
        Save the replay buffer to a Torch file.

        :param folder_path: Directory where the file will be saved.
        :param filename: Name of the file to save the buffer.
        """
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, filename)

        # Convert buffer to a list of tuples and handle numpy serialization
        buffer_list = [
            (torch.tensor(state, dtype=torch.float32),
             torch.tensor(target_policy, dtype=torch.float32),
             reward) for state, target_policy, reward in self.buffer
        ]
        torch.save(buffer_list, file_path)
        print(f"Saved buffer to {file_path} with size {len(buffer_list)}.")

    def load_buffer(self, folder_path, filename):
        """
        Load the replay buffer from a Torch file.

        :param folder_path: Directory where the file is located.
        :param filename: Name of the file to load the buffer.
        """
        file_path = os.path.join(folder_path, filename)

        # Load the buffer from a Torch file and handle numpy conversion
        try:
            buffer_list = torch.load(file_path)
            self.buffer = deque([
                (state.numpy() if isinstance(state, torch.Tensor) else state,
                 target_policy.numpy() if isinstance(target_policy, torch.Tensor) else target_policy,
                 reward) for state, target_policy, reward in buffer_list
            ], maxlen=self.buffer.maxlen)

            print(f"Loaded buffer from {file_path}.")

        except Exception as e:
            print(f"Error loading buffer from {file_path}. Exception: {e}. Starting with empty buffer.")


def create_and_store_replay_buffer_from_human_played_games(output_file=OUT_FILE):
    """
    Create a replay buffer from the Othello dataset.

    :param output_file:

    :returns: None
    """
    othello_data = read_othello_dataset()

    replay_buffer = ReplayBufferAlphaZero(capacity=2000000)

    experiences = []
    board = OthelloBoard(rows=8, cols=8)

    for idx, game in othello_data.iterrows():

        board.reset_board()

        game_actions = game['game_moves']

        if idx % 100 == 0 and idx != 0:
            print(f'Processed {idx} games ...')

        states, rewards, action_probs_list, players_list = [], [], [], []

        for i, move in enumerate(game_actions):

            player = board.next_player

            # If no valid actions available, set the rewards of all actions for the played game
            if len(board.get_valid_actions(player)) == 0:
                winner = board.check_winner()  # Determine the winner
                if winner == -1:  # Black wins
                    rewards = [1 if p == 0 else -1 for p in players_list]
                elif winner == -2:  # White wins
                    rewards = [1 if p == 1 else -1 for p in players_list]
                elif winner == -3:  # Draw
                    rewards = [0 for _ in players_list]
                break

            action = (ord(move[0]) - ord('A'), int(move[1]) - 1)
            
            action_one_hot_encoded = np.zeros((64,), dtype=np.float32)
            action_one_hot_encoded[action[0] * 8 + action[1]] = 1.0

            players_list.append(player)

            states.append(board.board.copy())
            action_probs_list.append(action_one_hot_encoded.reshape(64))

            rewards.append(0)

            player, _, _, _ = board.make_action(action, player)

        [replay_buffer.add(s, a, r) for s, a, r in zip(states, action_probs_list, rewards)]

    replay_buffer.save_buffer(folder_path='../../data', filename='replay_buffer_human.pth')


def load_replay_buffer(input_file=OUT_FILE, limit=float('inf')):
    """
    Load experiences from a JSON replay buffer file.

    :param limit:
    :param input_file: Path to the JSON file containing replay buffer data.

    :returns: ReplayBuffer instance containing the loaded experiences.
    """
    replay_buffer = ReplayBuffer()

    # Load experiences from the file
    if Path(input_file).exists():
        with open(input_file, 'r') as f:
            experiences = json.load(f)

        for idx, experience in enumerate(experiences):
            if idx >= limit:
                break

            state = np.array(experience['state'])
            action = tuple(experience['action'])
            reward = experience['reward']
            next_state = np.array(experience['next_state'])
            done = experience['done']

            # Add experience to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

        print(f"Loaded {len(replay_buffer)} experiences from {input_file}.")
    else:
        print(f"Input file {input_file} does not exist.")

    return replay_buffer


if __name__ == '__main__':
    create_and_store_replay_buffer_from_human_played_games()