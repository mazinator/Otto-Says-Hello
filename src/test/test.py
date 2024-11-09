"""
some test to verify that various stuff is actually behaving as expected
"""
from src.environment.board import OthelloBoard

print("Running tests..")

print("check default board at beginning..")
boardOk = OthelloBoard()
assert boardOk.cols == 8
assert boardOk.rows == 8

print("check some bad board configs..")
try:
    boardNok = OthelloBoard(cols=5, rows=5)
    assert False
except ValueError:
    assert True
try:
    boardNok = OthelloBoard(cols=8, rows=9)
    assert False
except ValueError:
    assert True
try:
    boardNok = OthelloBoard(cols=7, rows=8)
    assert False
except ValueError:
    assert True



print("All tests passed!")