class State ():

    def __init__ (self, size = 8, board = None, player = 1):
        self.player = player
        self.size = size
        self.board = []
        self.validMoves = []

        if board != None:
            for i in range (size):
                column = []
                for j in range (size):
                    column.append (board[i][j])
                self.board.append (column)
        else:
            for i in range (size):
                column = []
                for j in range (size):
                    if i == size / 2 - 1:
                        if j == size / 2:
                            column.append (1)
                            continue
                        if j == size / 2 - 1:
                            column.append (-1)
                            continue
                    
                    if i == size / 2:
                        if j == size / 2 - 1:
                            column.append (1)
                            continue
                        if j == size / 2:
                            column.append (-1)
                            continue

                    column.append (0)
                self.board.append (column)

        self.bc = 0
        self.wc = 0

        for i in range (size):
            for j in range (size):
                if self.board [i][j] == 1:
                    self.bc += 1
                    continue
                if self.board [i][j] == -1:
                    self.wc += 1

        self.count = (self.bc, self.wc)
        self.checkMoves ()

    def move (self, x, y):
        print ((x, y, self.player))
        found = False
        for i in range (len(self.validMoves)):
            (le_x, le_y, changes) = self.validMoves [i]
            if x == le_x and y == le_y:
                found = True
                break
#        (valid, changes) = self.isValid (x, y)
        if not found:
            print ("Invalid Move!!!")
            return None

        temp = []
        for i in range (self.size):
            column = []
            for j in range (self.size):
                column.append (self.board [i][j])
            temp.append (column)

        temp [x][y] = self.player
        print (x, y)
        for (x, y) in changes:
            print (x, y)
            temp [x][y] = self.player

        if self.player == 1:
            newPlayer = -1
        else:
            newPlayer = 1

        state = State (board = temp, player = newPlayer)
        return state

    def isValid (self, x, y):

        if x < 0 or y < 0 or x >= self.size or y >= self.size or self.board [x][y] != 0:
            return (False, [])

        directions = []
        for i in range (3):
            for j in range (3):
                directions.append ((i-1, j-1))

        directions.remove ((0, 0))

        converts = []
        for direction in directions:
            (x_change, y_change) = direction
            (valid, convert) = self.oneDirection (x, y, x_change, y_change)
            if not valid:
                continue
            for c in convert:
                converts.append (c)

        if (len (converts) == 0):
            return (False, converts)
        else:
            return (True, converts)

    def print (self):

        for j in range (self.size):
            s = ""
            for i in range (self.size):
                if self.board [i][j] == 0:
                    s = s + "."
                else:
                    if self.board [i][j] == -1:
                        s = s + "w"
                    else:
                        s = s + "b"
            print (s)

        s = ""
        for i in range (self.size):
            s = s + "-"

        print (s)
        return

    def printToFile (self, filename):

        outputFile = open (filename, 'a')        
        for j in range (self.size):
            s = ""
            for i in range (self.size):
                if self.board [i][j] == 0:
                    if i == self.size - 1:
                        s = s + " 0"
                    else:
                        s = s + " 0,"
                else:
                    if self.board [i][j] == -1:
                        if i == self.size - 1:
                            s = s + "-1"
                        else:
                            s = s + "-1,"
                    else:
                        if i == self.size - 1:
                            s = s + " 1"
                        else:
                            s = s + " 1,"
            s = s + "\n"
            outputFile.write (s)

        outputFile.close ()

        return

    def checkMoves (self):
        self.findValidMoves ()
        if len (self.validMoves) == 0:
            if self.player == 1:
                print ("No valid moves for black! It is now white's turn")
                self.player = -1
            else:
                print ("No valid moves for white! It is now black's turn")
                self.player = 1

            self.findValidMoves ()
            if len (self.validMoves) == 0:
                self.player = 0
                print ("The game has ended!!!!")
                if (self.bc > self.wc):
                    print ("Black has won by:")
                    print (self.count)
                else:
                    if (self.wc > self.bc):
                        print ("White has won by:")
                        print (self.count)
                    else:
                        print ("It is a tie of:")
                        print (self.count)

    #------------------------------------

    def oneDirection (self, x, y, x_change, y_change):
        defaultReturn = (False, [])
        new_x = x + x_change
        new_y = y + y_change

        if new_x < 0 or new_x >= self.size:
            return defaultReturn

        if new_y < 0 or new_y >= self.size:
            return defaultReturn

        if self.player == 1:
            opponent = -1
        else:
            opponent = 1

        if self.board[new_x][new_y] != opponent:
            return defaultReturn

        convert = []
        convert.append ((new_x, new_y))

        new_x += x_change
        new_y += y_change
        while (new_x > -1 and new_y > -1 and new_x < self.size and new_y < self.size):
            if self.board[new_x][new_y] == self.player:
                return (True, convert)
            else:
                if self.board[new_x][new_y] == 0:
                    return (False, [])
            convert.append ((new_x, new_y))
            new_x += x_change
            new_y += y_change

        return (False, [])

    def findValidMoves (self):
        for j in range (self.size):
            for i in range (self.size):
                (valid, converts) = self.isValid (i, j)
                if valid:
                    v = (i, j, converts)
                    self.validMoves.append (v)
