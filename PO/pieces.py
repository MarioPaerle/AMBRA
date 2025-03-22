"""Pieces and their Errors

ERRORS:
0 :  NO ERRORS
1 :  GOING OUT OF BOARD
2 :  GOING ON ANOTHER PIECE
"""
import pygame


class Piece:
    def __init__(self, pos, color=0):
        """Create a pawn at the given position"""
        self.pos = list(pos)
        self.image = None
        self.color = None
        if color == 0:
            self.color = (255, 255, 255)
        elif color == 1:
            self.color = (0, 0, 0)

        self.size = None

    def move(self, board, move):
        print(f"Move {move}")
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
            if self.pos[0] == 3:
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

    def render(self, game):
        # game.screen.blit(self.image, self.pos*game.size)
        pygame.draw.circle(game.screen, radius=self.size, color=self.color, center=((self.pos[0] + 0.5)*game.size, (self.pos[1] + 0.5)*game.size))


class Pawn(Piece):
    def __init__(self, pos, color=0):
        super().__init__(pos, color)
        self.type = 'pawn'
        self.size = 10

    def __str__(self):
        return f"{self.type} - {self.color}"

    def __repr__(self):
        return f"{self.type} - {self.color}"

class King(Piece):
    def __init__(self, pos, color=0):
        super().__init__(pos, color)
        self.type = 'king'
        self.size = 20

    def __str__(self):
        return f"{self.type} - {self.color}"

    def __repr__(self):
        return f"{self.type} - {self.color}"