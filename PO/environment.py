import pygame
import numpy as np

from PO.pieces import Pawn, King


class Environment:
    def __init__(self, name=''):
        self.renderer = None
        self.board = None

    def step(self, action):
        pass

    def reward(self):
        return None

    def render(self):
        self.renderer.render(self)

    def add_renderer(self, renderer):
        self.renderer = renderer


class Board:
    def __init__(self, name=''):
        """
        Re Bianco = 10
        Re Nero = -10
        Pedone Bianco = 1
        Pedone Nero = -1
        """
        self.name = name
        self.board = None
        self.errors = []
        self.pieces = []
        self.reset()

    def __getitem__(self, item):
        return self.board[*item]

    def render(self, game):
        self.create_board()
        for i in range(4):
            for j in range(4):
                rect = pygame.Rect(i*game.size, j*game.size, game.size, game.size)
                color = (139, 69, 19) if (i+j) % 2 == 0 else (222, 184, 135)
                if color == (255, 255, 255):
                    print(i, j)
                pygame.draw.rect(game.screen, color=color, rect=rect)

                piece = self.board[i, j]
                if piece != 0:
                    piece.render(game)

    def create_board(self):
        for piece in self.pieces:
            self.board[piece.pos[0], piece.pos[1]] = piece

    def reset(self):
        self.board = np.zeros((4, 4), dtype=object)
        rb = King((0, 0), 0)
        pb1 = Pawn((1, 0), 0)
        pb2 = Pawn((2, 0), 0)

        rn = King((3, 3), 1)
        pn1 = Pawn((2, 3), 1)
        pn2 = Pawn((1, 3), 1)
        self.pieces = [rb, pb1, pb2, rn, pn1, pn2]


class Game:
    def __init__(self, name=''):
        self.name = name
        self.screen = None
        self.board = Board()
        self.size = 100

    def update(self):
        self.render()

    def render(self):
        self.screen.fill((20, 20, 20))
        self.board.render(self)

    def reset(self):
        self.board.reset()


if __name__ == '__main__':
    b = Board()
    print(b.board)