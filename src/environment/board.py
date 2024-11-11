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
import copy


class OthelloBoard:
    def __init__(self, rows=10, cols=10, first_player=0):
        """
        In order to always have a "middle" to set the first 4 stones,
        the restrictions on those params have been chosen as:
        1. at least 6x6, at most 26x26 (mostly because of the alphabet and it's already more than enough)
        2. even numbers
        """
        self.board = None
        self.rows = rows
        self.cols = cols
        self.first_player_original = first_player
        self.next_player = self.first_player_original

        if (self.rows < 6 or self.rows % 2 != 0 or
                self.cols < 6 or self.cols % 2 != 0 or
                self.rows > 26 or self.cols > 26):
            raise ValueError("rows or cols smaller than 6, bigger than 26 or not even.")

        self.initial_board = None
        self.reset_board()

    def reset_board(self):
        # Set all fields to -1
        self.board = np.full((self.rows, self.cols), -1)

        # Set inner 4 disks for beginning
        self.board[self.rows // 2 - 1, self.cols // 2 - 1] = 1
        self.board[self.rows // 2, self.cols // 2] = 1
        self.board[self.rows // 2, self.cols // 2 - 1] = 0
        self.board[self.rows // 2 - 1, self.cols // 2] = 0

        # Save a copy of the initial board state
        self.initial_board = self.board.copy()

        self.next_player = self.first_player_original

    def is_initial_state(self):
        """
        Check if the board is in the initial state.
        """
        return np.array_equal(self.board, self.initial_board)

    def print_board(self):
        """
        Prints the board with letter column labels and digit row labels.
        """
        symbols = {-1: '-', 0: 'B', 1: 'W'}

        # Print the column labels with spacing
        column_labels = "   " + "  ".join(chr(ord('A') + i) for i in range(self.cols))
        print(column_labels)

        # Print each row with row number (1-based index), ensuring consistent alignment
        for row in range(self.rows):
            row_label = f"{row + 1:2} "  # Format row label to take up 2 spaces for alignment
            row_content = "  ".join(symbols[self.board[row, col]] for col in range(self.cols))
            print(row_label + row_content)

    def move_is_flipping_disks(self, row, col, player):
        """
        Checks if placing a disk at (row, col) is a flipping move for the player, which it has to do by the rules.
        """

        # Ensure the position is empty
        if self.board[row, col] != -1:
            raise ValueError("Position is not empty!")

        # Define opponent and directions for checking
        opponent = 1 if player == 0 else 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        # Check all directions to see if placing a disk would outflank opponent's disks
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.board[r, c] == opponent:
                    found_opponent = True
                elif self.board[r, c] == player:
                    if found_opponent:
                        return True  # Valid move as it can flip opponent's disks
                    else:
                        break
                else:
                    break
                r += dr
                c += dc

        return False  # No direction had a valid flip; return False

    def make_move(self, position, player):
        """
        Makes a move for the player if it's valid, placing a disk and flipping appropriate disks.

        Returns:
            next_player, board, valid_moves, winner
        """

        # Try decoding the given position, can be either e.g. A1 or (0,0)
        try:
            if isinstance(position, tuple) and len(position) == 2 and all(isinstance(i, int) for i in position):
                # Position is given as (row, col) with 0-based indexing
                row_idx, col_idx = position
            elif isinstance(position, str) and len(position) >= 2:
                # Position is given as a string like 'A1'
                col, row = position[0], position[1:]
                col_idx = ord(col.upper()) - ord('A')
                row_idx = int(row) - 1
            else:
                raise ValueError("Invalid input format. Use 'A1' or (0,0) with 0-based indexing.")
        except (IndexError, ValueError):
            raise ValueError("Invalid input format. Use format like 'A1' or (0,0) with 0-based indexing.")

        # Check if correct player is indeed playing
        if player is not self.next_player:
            player_name = "Black" if self.next_player == 0 else "White"
            raise ValueError(f"It's {player_name}'s turn!")

        # Check if move is inside the boundaries
        if not (0 <= col_idx < self.cols and 0 <= row_idx < self.rows):
            raise ValueError("Move is out of bounds.")

        # A move has to flip at least 1 disk
        if not self.move_is_flipping_disks(row_idx, col_idx, player):
            raise ValueError("Invalid move.")

        self.board[row_idx, col_idx] = player
        self.flip_disks(row_idx, col_idx, player)

        # Check if the other player can make a move, otherwise return same player
        otherPlayer = 1 if player == 0 else 0
        winner = 0  # only changed below if game's finished. Winner = 0 means that no one won, see check_winner()

        otherPlayer_valid_moves = self.get_valid_moves(otherPlayer)
        player_valid_moves = self.get_valid_moves(player)

        # Game is over, no more moves possible
        if len(otherPlayer_valid_moves) == 0 and len(player_valid_moves) == 0:
            winner = self.check_winner()
            valid_moves = []

        else:

            # If the other player has moves, they play next; otherwise, same player goes again
            if len(otherPlayer_valid_moves) > 0:
                valid_moves = otherPlayer_valid_moves
                self.next_player = otherPlayer

            else:
                valid_moves = player_valid_moves
                self.next_player = player

        return self.next_player, self.board, valid_moves, winner

    def flip_disks(self, row, col, player):
        """
        Flips the disks in all valid directions from (row, col) for the player.
        """

        opponent = 1 if player == 0 else 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

        for dr, dc in directions:
            disks_to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.board[r, c] == opponent:
                    disks_to_flip.append((r, c))
                elif self.board[r, c] == player:
                    for rr, cc in disks_to_flip:
                        self.board[rr, cc] = player
                    break
                else:
                    break
                r += dr
                c += dc

    def get_valid_moves(self, player):
        """
        Returns a list of valid moves for the given player.
        Each move is represented as a tuple (row, col).

        player: 0 for black, 1 for white
        """
        valid_moves = []

        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == -1 and self.move_is_flipping_disks(row, col, player):
                    valid_moves.append((row, col))

        return valid_moves

    def copy(self):
        return copy.deepcopy(self)

    def check_winner(self):
        black_count = np.sum(self.board == 0)
        white_count = np.sum(self.board == 1)

        if black_count > white_count:
            return -1
        elif white_count > black_count:
            return -2
        else:
            return -3

    def simulate_flip_count(self, position, player):
        """
        Simulates the move to count the number of stones that would be flipped without modifying the board.

        Returns The number of stones that would be flipped by this move.
        """
        row, col = position
        opponent = 1 if player == 0 else 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        flip_count = 0

        for dr, dc in directions:
            r, c = row + dr, col + dc
            temp_flip_count = 0
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.board[r, c] == opponent:
                    temp_flip_count += 1
                elif self.board[r, c] == player:
                    flip_count += temp_flip_count
                    break
                else:
                    break
                r += dr
                c += dc

        return flip_count

    def is_incomplete(self):
        """
        Checks if there are any empty spots on the board where no stones are placed.

        Returns:
            bool: True if there are empty spots, False otherwise.
        """
        for row in self.board:
            if -1 in row:
                return True
        return False
