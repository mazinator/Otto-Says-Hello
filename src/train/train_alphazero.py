from __future__ import division
from mcts.searcher.mcts import MCTS


import torch
import torch.nn as nn
import threading
import sys
from torch.nn.modules import loss
from torch.optim import Adam
import sys, time

from src.utils.mcts_wrapper import OthelloMCTS
from src.utils.nn_helpers import *
from torch.nn import functional as F
from src.utils.model_loader import *
from src.environment.board import *
from src.environment.replay_buffer import ReplayBufferAlphaZero
import math
from src.agents.alpha_zero import *

device = get_device()

def display_timer(start_time):
    while True:
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        # Red text using ANSI escape codes
        red_text = f"\033[91mElapsed Time: {hours:02}:{minutes:02}:{seconds:02}\033[0m"

        sys.stdout.write(f"\r{red_text}")
        sys.stdout.flush()
        time.sleep(10)


def train_agent(board, model, optimizer, episodes=sys.maxsize, batch_size=64, checkpoint_interval=500,
                model_loaded=False, episode_loaded=None, c=10e-4, simulations_between_training=50,
                epochs_for_batches=20, mcts_max_iterations=1000) -> None:
    """
    Trains an agent on selfplay.

    :param board: OthelloBoard, currently MUST be an 8x8 board
    :param model: Either AlphaZeroNet or AlphaZeroNetWithResiduals
    :param optimizer: .
    :param episodes: by default max integer, doesn't stop training
    :param batch_size: .
    :param checkpoint_interval: after this many episode, model will be saved.
    :param model_loaded: boolean if training is from scratch or not
    :param episode_loaded: the number of already trained episodes if model_loaded=True
    :param c: hyperparameter for the regularization term in the loss function
    :param simulations_between_training: the maximum time for the MCTS simulator package
    :param epochs_for_batches: the number of batches to sample from the replay buffer.
    :param mcts_max_iterations: hyperparameter for MCTS
    """

    timer_thread = threading.Thread(target=display_timer, args=(time.time(),), daemon=True)
    timer_thread.start()

    model.to(device)
    start = time.time()
    start_episode = episode_loaded if model_loaded else 0
    loss, policy_loss, value_loss, regularization_term = 'TBD', None, None, None
    replay_buffer = ReplayBufferAlphaZero()

    for episode in range(start_episode, episodes):

        board.reset_board()

        # Some print statements
        if episode % 100 == 0:
            print(
                f'Running episode: {episode} from {episodes}, running since {round((time.time() - start) / 60, 3)} minutes'
            )

        states, rewards, action_probs_list, players_list = [], [], [], []

        # Play as long as the game does not end
        while True:
            # Do Monte Carlo Tree Search from current board state and derive action probabilities
            player = board.next_player

            # If no valid actions available, set the rewards of all action
            # taken to the last reward (1/-1/0 if win/loss/tie)
            if len(board.get_valid_actions(player)) == 0:
                winner = board.check_winner()  # Determine the winner
                if winner == -1:  # Black wins
                    rewards = [1 if p == 0 else -1 for p in players_list]
                elif winner == -2:  # White wins
                    rewards = [1 if p == 1 else -1 for p in players_list]
                elif winner == -3:  # Draw
                    rewards = [0 for _ in players_list]
                break

            initial_state = OthelloMCTS(board, player=player, model=model)

            # Initialize MCTS
            searcher = MCTS(time_limit=mcts_max_iterations)

            searcher.search(initial_state=initial_state)

            # Get the policy distribution from the AlphaZero model
            action_probs = initial_state.get_policy_distribution()

            players_list.append(player)

            best_action_index = np.argmax(action_probs)  # Index of the maximum probability
            best_action = tuple(map(int, divmod(best_action_index, 8)))

            # Store state (board configuration) and action taken in each state
            # Note: I don't know how they did it in the original paper, but action here is
            # just the index in the list of available action. This should not be a problem,
            # as long as the list of actions returned by the board for a given state are
            # deterministic, which they are currently. If one changes this behaviour,
            # this would completely break training!
            states.append(board.board.copy())
            action_probs_list.append(action_probs.reshape(64))

            # Placeholder for rewards during self-play, will be changed to actual rewards
            # later in the if-statement above.
            rewards.append(0)

            # Perform the chosen action (check out explanation 10 lines above, this is only
            # valid as get_valid_action() is deterministic!
            player, _, _, _ = board.make_action(best_action, player)

        # Add experience from the played round into the replay buffer
        [replay_buffer.add(s, a, r) for s, a, r in zip(states, action_probs_list, rewards)]

        if episode % simulations_between_training == 0 and episode is not start_episode:
            print(f'Starting to train {epochs_for_batches} batches after {episode-start_episode+1} episodes. Buffer '
                  f'has length {len(replay_buffer)}.')

            for i in range(epochs_for_batches):

                # Train the model using batches
                batch = replay_buffer.sample(batch_size)
                batch_states, batch_action_probs, batch_rewards = zip(*batch)

                # Prepare the input tensor for the model, see description of prepare_alphazero_board_tensor()
                model_input_tensor = torch.stack([
                    prepare_alphazero_board_tensor(state, board.next_player, device).squeeze(0) for state in batch_states
                ]).to(device)

                # Forward pass
                optimizer.zero_grad()
                p, v = model(model_input_tensor)

                # Compute loss for the batch
                policy_loss = F.kl_div(input=p,
                                       target=torch.tensor(batch_action_probs, device=device, dtype=torch.float32),
                                       reduction='batchmean')
                mse_input = v.squeeze()
                mse_target = torch.tensor(batch_rewards, device=device, dtype=torch.float32)
                value_loss = F.mse_loss(input=mse_input,
                                        target=mse_target,
                                        reduction='sum')
                regularization_term = c * sum(torch.norm(param) ** 2 for param in model.parameters())
                loss = policy_loss + value_loss + regularization_term

                # Print current loss and how it is distributed across the 3 parts
                print(
                    f'Total loss: {loss:.4f}. Policy loss: {policy_loss:.4f}. Value loss: {value_loss:.4f}. '
                    f'Regularization term: {regularization_term:.4f}')

                # Backward pass + optimization
                loss.backward()
                optimizer.step()

        # Create checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, episode + 1)


if __name__ == '__main__':
    # Load infrastructure
    board = OthelloBoard(8, 8)
    model = AlphaZeroNet(8, 8)
    model, episode = load_model('cp_alphazero_0.001_lr', model)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    train_agent(board, model, optimizer, model_loaded=True, episode_loaded=episode,
                checkpoint_interval=50, batch_size=32, epochs_for_batches=30,
                mcts_max_iterations=2000, simulations_between_training=40)