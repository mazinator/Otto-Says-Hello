import torch
import torch.mps
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math
import os
import re
import sys

from src.environment.board import OthelloBoard


class AlphaZeroNet(nn.Module):
    def __init__(self, rows, cols):
        super(AlphaZeroNet, self).__init__()
        self.rows = rows
        self.cols = cols

        # Convolutional layers
        # Note: the 3 channels are necessary because AlphaZero architecture
        # actually has 3 inputs of board sizes: Player's Pieces, Opponent's Pieces, Empty places
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * rows * cols, rows * cols)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board):
        # Convert board to tensor and add batch and channel dimensions
        x = board.view(-1, 3, self.rows, self.cols).float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)  # Flatten
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class MCTS:
    def __init__(self, board, model, simulations=800):
        self.board = board
        self.model = model
        self.simulations = simulations
        self.visits = {}
        self.values = {}
        self.priors = {}
        self.children = {}

    def search(self, node, player, device):
        if node not in self.visits:
            # Expand the node
            valid_moves = self.board.get_valid_actions(player)
            if not valid_moves:
                return -self.evaluate(node, player, device)

            self.children[node] = valid_moves
            board_tensor = prepare_board_tensor(self.board.board, player).to(device)
            p, v = self.model(board_tensor)
            self.priors[node] = p.squeeze().detach().numpy()
            self.values[node] = 0
            self.visits[node] = 0
            return -v.item()

        best_ucb = -float("inf")
        best_move = None

        # Use Upper Confidence Bound for Trees (UCT)
        for move in self.children[node]:
            q = self.values.get((node, move), 0)
            n = self.visits.get((node, move), 0)
            prior = self.priors[node][move]
            ucb = q + 1.0 * prior * math.sqrt(self.visits[node]) / (1 + n)

            if ucb > best_ucb:
                best_ucb = ucb
                best_move = move

        # Make the move
        if best_move is not None:
            new_board = self.board.copy()
            new_board.make_action(best_move, player)
            result = -self.search(new_board, 1 - player, device)

            # Update visit count and value
            self.visits[(node, best_move)] = self.visits.get((node, best_move), 0) + 1
            self.visits[node] += 1
            self.values[(node, best_move)] = (
                    (self.values.get((node, best_move), 0) + result) / self.visits[(node, best_move)]
            )

            return result
        else:
            # No valid moves, return evaluation
            return -self.evaluate(node, player, device)

    def evaluate(self, node, player, device):
        # Neural network evaluation
        board_tensor = prepare_board_tensor(self.board.board, player).to(device)
        _, v = self.model(board_tensor)
        return v.item()

    def get_policy(self, tau=1):
        board_key = self.board.get_board_state_key()  # Get an immutable representation of the board state
        visit_counts = [self.visits.get((board_key, move), 0) for move in self.children.get(board_key, [])]
        if len(visit_counts) == 0:
            # No valid moves available, return uniform policy for all possible moves
            valid_moves = self.board.get_valid_actions(self.board.next_player)
            if not valid_moves:
                return []
            policy = [1 / len(valid_moves)] * len(valid_moves)
            return policy

        if tau == 0:
            best_move = np.argmax(visit_counts)
            policy = [0] * len(visit_counts)
            policy[best_move] = 1
        else:
            visit_counts = [count ** (1 / tau) for count in visit_counts]
            total_count = sum(visit_counts)
            policy = [count / total_count for count in visit_counts]

        return policy


def train_agent(board, model, optimizer, device, episodes=sys.maxsize, batch_size=64, checkpoint_interval=50000):
    model.to(device)
    import time as t
    start = t.time()
    for episode in range(episodes):
        board.reset_board()

        if episode % 1000 == 0:
            print(f'Running episode: {episode} from {episodes}, running since {round((t.time()-start)/60, 3)} minutes')

        # Self-play to generate data
        states, actions, rewards = [], [], []
        for _ in range(100):  # Simulate games
            mcts = MCTS(board.copy(), model)
            player = board.next_player
            action_probs = mcts.get_policy()

            if len(action_probs) == 0:
                # No valid actions available (likely a terminal state)
                final_reward = board.check_winner()
                rewards = [final_reward for _ in rewards]
                break

            action = np.random.choice(len(action_probs), p=action_probs)

            states.append(board.board.copy())
            actions.append(action)
            rewards.append(0)  # Placeholder for rewards during self-play

            # Make the chosen action
            valid_moves = board.get_valid_actions(player)
            board.make_action(valid_moves[action], player)

        # Assign rewards based on the game's outcome
        final_reward = board.check_winner()
        rewards = [final_reward for _ in rewards]

        # Train the model using batches
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]

            # Prepare the batch tensor
            batch_tensors = torch.stack([
                prepare_board_tensor(state, board.next_player, device).squeeze(0) for state in batch_states
            ]).to(device)

            # Forward pass
            optimizer.zero_grad()
            p, v = model(batch_tensors)

            # Compute the loss for the batch
            policy_loss = -torch.stack([
                torch.log(p[j, batch_actions[j]]) * batch_rewards[j] for j in range(len(batch_states))
            ]).sum()

            value_loss = ((v.squeeze() - torch.tensor(batch_rewards, device=device)) ** 2).sum()
            loss = policy_loss + value_loss

            # Backward pass
            loss.backward()
            optimizer.step()

        if (episode + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, episode + 1)

def prepare_board_tensor(board, player, device='cpu'):
    # Create a tensor with 3 channels: current player's pieces, opponent's pieces, and empty spots
    current_player_layer = (board == player).astype(np.float32)
    opponent_player_layer = (board == (1 - player)).astype(np.float32)
    empty_layer = (board == -1).astype(np.float32)
    board_tensor = np.array([current_player_layer, opponent_player_layer, empty_layer])
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    return board_tensor


def save_checkpoint(model, optimizer, episode, path="../../.checkpoints"):
    """Save a checkpoint of the model and optimizer."""
    if not os.path.exists(path):
        os.makedirs(path)  # Create directory if it doesn't exist
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, os.path.join(path, f"cp_alphazero_{optimizer.param_groups[0]['lr']}_lr_{episode}_episodes.pth"))
    print(f"Checkpoint saved for episode {episode}.")


def get_checkpoint(prefix, folder_path, episode_number=None):
    """
    Get the latest checkpoint file based on the highest episode number or a specific episode.

    Args:
        prefix (str): The prefix of the checkpoint file (e.g., "cp_alphazero_0.001_lr").
        folder_path (str): The folder path to search for checkpoint files.
        episode_number (int, optional): Specific episode number to search for. Default is None.

    Returns:
        str: The path to the latest checkpoint file, or None if no matching file is found.
    """
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        return None

    # Pattern to extract episode numbers from filenames
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_episodes\.pth")

    latest_file = None
    max_episode = -1

    for file in os.listdir(folder_path):
        if file.startswith(prefix) and file.endswith(".pth"):
            match = pattern.search(file)
            if match:
                current_episode = int(match.group(1))  # Extract episode number from the filename
                if episode_number is not None:
                    # Return the specific episode file if found
                    if current_episode == episode_number:
                        return os.path.join(folder_path, file)
                elif current_episode > max_episode:
                    # Update the latest file based on the highest episode number
                    max_episode = current_episode
                    latest_file = os.path.join(folder_path, file)

    if latest_file:
        return latest_file if episode_number is None else None


if __name__ == '__main__':

    #cp = get_checkpoint('checkpoint_alphazero_0.001_lr', "../../.checkpoints")
    board = OthelloBoard(8, 8)
    model = AlphaZeroNet(board.rows, board.cols)
    optimizer = Adam(model.parameters(), lr=0.001)
    device_type = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Train the model
    train_agent(board, model, optimizer, device_type)
