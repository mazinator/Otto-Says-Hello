import os.path
from pathlib import Path

filename = input ("What is the filename?")
fobj = open(filename, "r")
biglist = list ()
for line in fobj:
	if len (line) <= 1:
		continue
	if line[0] == "/":
		continue

	size = len(line)/2
	templist = list ()
	for i in range (int(size)):
		index = 2 * i
		index1 = index + 1
		x = ord (line[index])
		if x < ord ('a'):
			x = x - ord ('A')
		else:
			x = x - ord ('a')
		y = int (line [index1]) - 1
		templist.append ((x, y))
	biglist.append (templist)

currentDirectory = os.path.dirname(os.path.abspath(__file__))
counter = 0
outputfilename = "record_" + str(counter)
outputfilename = currentDirectory + "/" + outputfilename
outfile = Path (outputfilename)

for game in biglist:
	while (outfile.is_file()):
		counter += 1
		outputfilename = "record_" + str(counter)
		outputfilename = currentDirectory + "/" + outputfilename
		outfile = Path (outputfilename)

	outputFile = open (outputfilename, 'w')
	for line in game:
		outputFile.write (' '.join (map (str, line)))
		outputFile.write ("\n")

outputFile.close ()