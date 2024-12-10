import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.model_loader import *
from src.utils.nn_helpers import prepare_alphazero_board_tensor


class AlphaZeroNet(nn.Module):
    def __init__(self, rows, cols):
        """
        Rather simple implementation without residual blocks, used in the beginning because it was easier to debug

        :param rows: int
        :param cols: int
        """
        super(AlphaZeroNet, self).__init__()
        self.rows = 8  # rows fixed currently!
        self.cols = 8  # cols fixed currently!

        # Convolutional layers
        # Note: the 3 channels are necessary because AlphaZero architecture
        # actually has 3 inputs of size board-size: Player's Pieces, Opponent's Pieces, Empty places
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Policy head; probability distribution over all possible actions at the current state
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * rows * cols, rows * cols)

        # Value head; expected outcome of the game from the current state for the current player
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board_tensor):

        # Convert board to tensor and add batch and channel dimensions
        x = board_tensor.view(-1, 3, self.rows, self.cols).float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head; probability distribution over all possible actions at the current state
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head; expected outcome of the game from the current state for the current player
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # shape(p) = (1,64)
        # shape(v) = (1,1)
        return p, v

    def get_action(self, player, board):

        print(f"AlphaZero ({['black', 'white'][player]}) is thinking...")
        board_tensor = prepare_alphazero_board_tensor(board.board, player)
        with torch.no_grad():
            policy, _ = self.forward(board_tensor)
        valid_moves = board.get_valid_actions(player)
        move_probs = torch.exp(policy).cpu().numpy().flatten()
        return max(valid_moves, key=lambda mv: move_probs[mv[0] * board.cols + mv[1]])


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Residual Block class for more sophisticated AlphaZero implementation.

        :param channels:
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AlphaZeroNetWithResiduals(nn.Module):
    def __init__(self, rows, cols, num_residual_blocks=5):
        """
        Original AlphaZero architecture.

        :param rows: int
        :param cols: int
        :param num_residual_blocks: int
        """
        super(AlphaZeroNetWithResiduals, self).__init__()
        self.rows = rows
        self.cols = cols

        # Initial convolutional layer
        self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(128)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(128) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, rows * cols)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board):

        # Ensure the board is properly shaped
        x = board.view(-1, 3, self.rows, self.cols).float()

        # Initial convolutional layer
        x = F.relu(self.bn(self.conv(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
