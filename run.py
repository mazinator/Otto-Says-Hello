from src.test.test import *
from src.utils.data_loader import *
import argparse


def main(run_tests=False):
    if run_tests: run_all_tests()

    othello_games = read_othello_dataset()
    print(othello_games)


if __name__ == '__main__':
    main()