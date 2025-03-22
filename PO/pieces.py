"""Pieces and their Errors

ERRORS:
0 :  NO ERRORS
1 :  GOING OUT OF BOARD
2 :  GOING ON ANOTHER PIECE
"""
class Piece:
    def __init__(self, pos, color=0):
        """Create a pawn at the given position"""
        self.pos = []

    def move(self, board, move):
        """Move The piece of one step in the given direction (clockwise starting from the top)"""
        if move == 0:
            if self.pos[1] == 3:
                board.errors.append(1)
                return 1
            elif board[self.pos[0], self.pos[1] + 1] != 0:
                board.errors.append(2)
                return 2
            self.pos[1] += 1

        elif move == 1:
            if self.pos[0] == 0:
                board.errors.append(1)
                return 1
            elif board[self.pos[0] + 1, self.pos[1]] != 0:
                board.errors.append(2)
                return 2
            self.pos[0] += 1

        elif move == 2:
            if self.pos[1] == 0:
                board.errors.append(1)
                return 1
            elif board[self.pos[0], self.pos[1] - 1] != 0:
                board.errors.append(2)
                return 2
            self.pos[1] -= 1

        elif move == 3:
            if self.pos[0] == 0:
                board.errors.append(1)
                return 1
            elif board[self.pos[0] - 1, self.pos[1]] != 0:
                board.errors.append(2)
                return 2
            self.pos[0] -= 1


class Pawn(Piece):
    def __init__(self, pos, color=0):
        super().__init__(pos, color)
        self.type = 'pawn'

class King(Piece):
    def __init__(self, pos, color=0):
        super().__init__(pos, color)
        self.type = 'king'