from src.environment.board import OthelloBoard

def main():
    othello_board = OthelloBoard()
    othello_board.make_move("D3", 1)
    othello_board.print_board()









if __name__ == '__main__':
    main()