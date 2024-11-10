import os.path
from pathlib import Path
from gamestate import *

#counter = 0

for counter in range (50):
	currentDirectory = os.path.dirname(os.path.abspath(__file__))
	inputfilename = currentDirectory + "/" + "record_" + str(counter)
	infile = open (inputfilename, "r")

	moveRecord = ""

	state = State ()
	for line in infile:

		li = line.split ()
		x = li[0]
		y = li[1]

		if state.player == 1:
			moveRecord = moveRecord + "1 "
		else:
			moveRecord = moveRecord + "0 "

		moveRecord = moveRecord + x + " " + y + "\n"

		state = state.move (int(x), int(y))

	infile.close ()
	infile = open (inputfilename, "w")
	infile.write (moveRecord)
	infile.close ()



