"""This part is used to create a more precise ReplayBuffer in parallel to training the model
on a ReplayBuffer with a smaller iteration_time on the MCTS.
"""

from torch.optim.adam import Adam

from src.agents.alpha_zero import AlphaZeroNetWithResiduals
from src.environment.board import OthelloBoard
from src.train.train_alphazero import *
from src.utils.model_loader import load_model

if __name__ == '__main__':
    learning_rate = 0.001
    learning_rate_load_from = 0.001

    # Load Othello-Board
    board = OthelloBoard(8, 8)

    # Load neural network model
    model = AlphaZeroNetWithResiduals(8, 8)

    # Try to retrieve past model with given learning rate
    model, episode = load_model(f'cp_alphazero_residuals_{learning_rate_load_from}_lr', model)

    # Set optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Multiple further (hyper)parameters needed for training
    model_loaded = True
    episode_loaded = episode
    checkpoint_interval = 20
    batch_size = 32
    epochs_for_batches = 15
    mcts_max_time = 5000
    simulations_between_training = 150
    mcts_exploration_constant = 4
    replay_buffer_in = 'replay_buffer_alphazero_5000.pth'
    replay_buffer_out = 'replay_buffer_alphazero_5000.pth'
    replay_buffer_folder_path = '../../data/'
    mcts_only = True

    train_agent(board, model, optimizer, model_loaded=model_loaded, episode_loaded=episode_loaded,
                checkpoint_interval=checkpoint_interval, batch_size=batch_size, epochs_for_batches=epochs_for_batches,
                mcts_max_time=mcts_max_time, simulations_between_training=simulations_between_training,
                mcts_exploration_constant=mcts_exploration_constant, replay_buffer_in=replay_buffer_in,
                replay_buffer_out=replay_buffer_out, replay_buffer_folder_path=replay_buffer_folder_path,
                mcts_only=mcts_only)
