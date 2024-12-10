import torch
import os
import re

from src.agents.alpha_zero import *
from src.utils.nn_helpers import *


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
    torch.save(checkpoint,
               os.path.join(path, f"cp_alphazero_{optimizer.param_groups[0]['lr']}_lr_{episode}_episodes.pth"))
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
        :param model_loaded:
    """
    if not os.path.exists(folder_path):
        #print(f"No pretrained model found under: {folder_path}. Starting from zero")
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
                        return os.path.join(folder_path, file), episode_number
                elif current_episode > max_episode:
                    # Update the latest file based on the highest episode number
                    max_episode = current_episode
                    latest_file = os.path.join(folder_path, file)

    if latest_file:
        return latest_file, max_episode if episode_number is None else None
    else:
        return None, None


def load_model(prefix, model, folder_path='../../.checkpoints'):
    device = get_device()

    try:
        checkpoint_path, episode = get_checkpoint(prefix, folder_path)
    except:
        print(f"No pretrained model found for prefix: {prefix} in folder {folder_path}. Starting from zero")
        return model, 1

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded AlphaZero model from episode {checkpoint['episode']}")
        return model, episode
    else:
        print(f"No pretrained model found for prefix: {prefix}. Starting from zero")
        return model, 1
