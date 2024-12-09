import torch
import numpy as np


def get_device():
    """
    Returns the available device.
    """
    #return 'cpu'
    if torch.backends.cudnn.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device("cpu")


def prepare_alphazero_board_tensor(board, player, device=get_device()):
    """
    Create a tensor with 3 channels: current player's pieces, opponent's pieces, and empty spots

    :param board: OthelloBoard instance
    :param player: 0 or 1 indicating current player
    :param device:
    :return: Board tensor
    """
    current_player_layer = (board == player).astype(np.float32)
    opponent_player_layer = (board == (1 - player)).astype(np.float32)
    empty_layer = (board == -1).astype(np.float32)
    board_tensor = np.array([current_player_layer, opponent_player_layer, empty_layer])
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    return board_tensor