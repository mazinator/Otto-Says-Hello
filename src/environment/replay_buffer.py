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

from src.environment.board import OthelloBoard
from src.utils.data_loader import read_othello_dataset
from pathlib import Path

OUT_FILE = 'data/replay_buffer.json'


class ReplayBuffer:
    def __init__(self, capacity=2000000):
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
    def __init__(self, capacity=20000):
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
        print(f"Saved buffer to {file_path}.")

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




def create_and_store_replay_buffer(output_file=OUT_FILE):
    """
    Create a replay buffer from the Othello dataset.

    :param output_file:

    :returns: None
    """
    othello_data = read_othello_dataset()

    experiences = []

    for idx, game in othello_data.iterrows():

        board = OthelloBoard(rows=8, cols=8)
        moves = game['game_moves']

        if idx % 200 == 0 and idx != 0:
            print(f'Processed {idx} games ...')

        for i, move in enumerate(moves):
            state = board.board.tolist()
            player = board.next_player

            # Parse and execute move
            action = (ord(move[0]) - ord('A'), int(move[1]) - 1)
            next_player, next_board, valid_actions, winner = board.make_action(action, player)

            # Check reward
            reward = 0
            if i == len(moves) - 1:
                if winner == -1:
                    reward = 1 if player == 0 else -1
                elif winner == -2:
                    reward = 1 if player == 1 else -1
                elif winner == -3:
                    reward = 0.5

            winner_found = winner != 0

            # Store experience as a dictionary
            experiences.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_board.tolist(),
                "done": winner_found
            })

            if winner_found:
                break

    # Check if file exists, otherwise create
    if not Path(OUT_FILE).exists():
        Path(OUT_FILE).touch()
    else:
        Path(OUT_FILE).unlink()
        Path(OUT_FILE).touch()


    # Save all experiences to the output file
    with open(output_file, 'w') as f:
        json.dump(experiences, f)

    print(f"Saved {len(experiences)} experiences to {output_file}.")


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
