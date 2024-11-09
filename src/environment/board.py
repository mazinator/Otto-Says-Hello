"""
This file contains the Othello-Board.

It is by default a 8x8 matrix, therefore there are 64 positions
on the board. Each position can have max. 1 disk on it. A disk
can be flipped according to the rules.

Each position on the board can have 3 different states:
-1: empty
0: black disk
1: white disk

A small recap of the rules:
1. Black always moves first
2. If a player cannot outflank and flip at least one opposing disk,
    the opponent moves again. If a move is available, a player has
    to play.
3. In simple language, you can flip disks horizontally/vertically/diagonally.
4. Players may not skip over their own color disk(s) to outflank an opposing disk.
5. All disks that can be flipped, have to be flipped
6. When it is no longer possible for either player to move, the game is over. Disks are
    counted and the player with the majority of their color showing is the winner.
"""

import numpy as np


class OthelloBoard:
    def __init__(self, rows=8, cols=8):
        """
        In order to always have a "middle" to set the first 4 stones,
        the restrictions on those params have been chosen as:
        1. at least 6x6
        2. even numbers
        """
        self.board = None
        self.rows = rows
        self.cols = cols

        if self.rows < 6 or self.rows % 2 != 0 or self.cols < 6 or self.cols % 2 != 0:
            raise ValueError("rows or cols smaller than 6 or not even.")

        self.reset_board()

    def reset_board(self):

        # Set all fields to -1
        self.board = np.full((self.rows, self.cols), -1)

        # Set inner 4 disks for beginning
        self.board[self.rows // 2 - 1, self.cols // 2 - 1] = 0
        self.board[self.rows // 2, self.cols // 2] = 0
        self.board[self.rows // 2, self.cols // 2 - 1] = 1
        self.board[self.rows // 2 - 1, self.cols // 2] = 1

    def print_board(self):
        """
        Prints the board with A-H column labels and 1-8 row labels.
        """
        symbols = {-1: '-', 0: 'B', 1: 'W'}
        column_labels = "  " + " ".join(chr(ord('A') + i) for i in range(self.cols))
        print(column_labels)  # Print the column headers A-H

        for row in range(self.rows):
            # Print each row with row number (1-based index)
            row_label = f"{row + 1} "  # Adjust row index to 1-based
            row_content = " ".join(symbols[self.board[row, col]] for col in range(self.cols))
            print(row_label + row_content)

    #def make_move(self):
