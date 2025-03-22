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
        self.ended = False
        self.winner = None
        self.reset()

    def __getitem__(self, item):
        return self.board[*item]

    def update(self):
        rb = self.pieces[0]
        neighbours = neighbour(rb.pos)
        for n in neighbours:
            if n[0] < 0 or n[1] < 0 or n[0] > 3 or n[1] > 3:
                continue
            casel = self.board[n]
            if casel in (self.pieces[4], self.pieces[5]):
                self.ended = True
                self.winner = 1

        rn = self.pieces[3]
        neighbours = neighbour(rn.pos)
        for n in neighbours:
            if n[0] < 0 or n[1] < 0 or n[0] > 3 or n[1] > 3:
                continue
            casel = self.board[n]
            if casel in (self.pieces[1], self.pieces[2]):
                self.ended = True
                self.winner = 0

        self.create_board()


    def render(self, game):
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
        self.board = np.zeros((4, 4), dtype=object)
        for piece in self.pieces:
            self.board[piece.pos[0], piece.pos[1]] = piece

    def reset(self):
        self.errors = []
        self.pieces = []
        self.ended = False
        self.winner = None
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
        self.do_render = True
        self.current_player = 0

        # PC PLAYING
        self.current_case = [0, 0]

    def update(self, events=None):
        self.board.update()
        if self.do_render:
            self.render()
            self.check_events(events)
        if self.board.ended:
            print(f"Game ended, winner: {self.board.winner}")

    def render(self):
        self.screen.fill((20, 20, 20))
        self.board.render(self)

        t1 = Text(f"Current Player: {self.current_player}", (5, 450))
        t1.render(self.screen)

        pygame.draw.rect(self.screen, (0, 100, 255),
                         (self.current_case[0] * self.size, self.current_case[1] * self.size, self.size, self.size),
                         3)  # width = 3

    def reset(self):
        self.board.reset()

    def execute(self, action):
        if self.current_player == 0 and action >= 12:
            print(f"Warning, moving a piece of the other player: {action}")
        elif self.current_player == 1 and action < 12:
            print(f"Warning, moving a piece of the other player: {action}")
        else:
            self.current_player = 1 - self.current_player

        piece = self.board.pieces[action // 4]
        if action >= 12:
            move = (action + 2) % 4
        else:
            move = action % 4
        piece.move(self.board, move)

    def check_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                piece = self.board[self.current_case[0], self.current_case[1]]
                if piece != 0:
                    if piece.color == (255, 255, 255):
                        if event.key == pygame.K_UP:
                            self.execute(self.board.pieces.index(piece) * 4 + 0)
                            # piece.move(self.board, 0)
                        elif event.key == pygame.K_RIGHT:
                            self.execute(self.board.pieces.index(piece) * 4 + 1)
                            # piece.move(self.board, 1)
                        elif event.key == pygame.K_DOWN:
                            self.execute(self.board.pieces.index(piece) * 4 + 2)
                            # piece.move(self.board, 2)
                        elif event.key == pygame.K_LEFT:
                            self.execute(self.board.pieces.index(piece) * 4 + 3)
                            # piece.move(self.board, 3)
                    else:
                        print('asdadsas')
                        if event.key == pygame.K_UP:
                            self.execute(self.board.pieces.index(piece) * 4 + 0)
                            # piece.move(self.board, 0)
                        elif event.key == pygame.K_RIGHT:
                            self.execute(self.board.pieces.index(piece) * 4 + 3)
                            # piece.move(self.board, 1)
                        elif event.key == pygame.K_DOWN:
                            self.execute(self.board.pieces.index(piece) * 4 + 2)
                            # piece.move(self.board, 2)
                        elif event.key == pygame.K_LEFT:
                            self.execute(self.board.pieces.index(piece) * 4 + 1)
                            # piece.move(self.board, 3)

                if event.key == pygame.K_w:
                    self.current_case[1] = (self.current_case[1] - 1) % 4
                elif event.key == pygame.K_s:
                    self.current_case[1] = (self.current_case[1] + 1) % 4
                elif event.key == pygame.K_a:
                    self.current_case[0] = (self.current_case[0] - 1) % 4
                elif event.key == pygame.K_d:
                    self.current_case[0] = (self.current_case[0] + 1) % 4

class Text:
    def __init__(self, content, pos, font_size=24, color=(255, 255, 255)):
        self.content = content
        self.pos = pos
        self.font_size = font_size
        self.color = color
        self.font = pygame.font.Font(None, self.font_size)

    def render(self, screen):
        text_surface = self.font.render(self.content, True, self.color)
        screen.blit(text_surface, self.pos)


def neighbour(pos):
    return [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]), (pos[0], pos[1]+1), (pos[0], pos[1]-1)]


if __name__ == '__main__':
    b = Board()
    print(b.board)