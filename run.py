from src.agents.medium_agents import *
from src.test.test import *
from src.utils.data_loader import *
import argparse


def main(run_tests=False):

    # run tests at the beginning if desired
    if run_tests:
        run_all_tests()

    #othello_games = read_othello_dataset()
    #print(othello_games)

    environment = OthelloBoard()
    agent = SimpleQLearningAgent(environment)

    # Train the Q-Learning Agent over multiple episodes
    for episode in range(1000):
        agent.play_episode()
        agent.decay_epsilon(0.9995)





if __name__ == '__main__':
    main()
