import pygame
import numpy as np

from PO.pieces import Pawn, King
import time


class Board:
    def __init__(self, name='', dimension=5):
        """
        Re Bianco = 10
        Re Nero = -10
        Pedone Bianco = 1
        Pedone Nero = -1
        """
        self.name = name
        self.board = None
        self.dimensions = dimension
        self.state = np.zeros((self.dimensions, self.dimensions), dtype=object)
        self.errors = [0]
        self.pieces = []
        self.ended = False
        self.winner = None
        self.reset()

    def __getitem__(self, item):
        return self.board[*item]

    def update(self):
        """re bianco, controlla se il re bianco si trova vicino ad un pedone nero"""
        rb = self.pieces[0]
        neighbours = neighbour(rb.pos)
        for n in neighbours:
            if n[0] < 0 or n[1] < 0 or n[0] > self.dimensions - 1 or n[1] > self.dimensions - 1:
                continue
            casel = self.board[n]
            if casel in (self.pieces[4], self.pieces[5]):
                self.ended = True
                self.winner = 1
        """re nero, controlla se il re nero si trova vicino ad un pedone bianco"""
        rn = self.pieces[3]
        neighbours = neighbour(rn.pos)
        for n in neighbours:
            if n[0] < 0 or n[1] < 0 or n[0] > self.dimensions - 1 or n[1] > self.dimensions - 1:
                continue
            casel = self.board[n]
            if casel in (self.pieces[1], self.pieces[2]):
                self.ended = True
                self.winner = 0

        self.create_board()

    def render(self, game):
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                rect = pygame.Rect(i*game.size, j*game.size, game.size, game.size)
                color = (139, 69, 19) if (i+j) % 2 == 0 else (222, 184, 135)
                pygame.draw.rect(game.screen, color=color, rect=rect)

                piece = self.board[i, j]
                if piece != 0:
                    piece.render(game)

    def create_board(self):
        self.board = np.zeros((self.dimensions, self.dimensions), dtype=object)
        self.state = np.zeros((self.dimensions, self.dimensions), dtype=np.int8)
        for piece in self.pieces:
            self.board[piece.pos[0], piece.pos[1]] = piece
            self.state[piece.pos[0], piece.pos[1]] = piece.value


    def reset(self):
        self.errors = [0]
        self.pieces = []
        self.ended = False
        self.winner = None
        self.board = np.zeros((self.dimensions, self.dimensions), dtype=object)
        rb = King((0, 0), 0)
        pb1 = Pawn((1, 0), 0)
        pb2 = Pawn((2, 0), 0)

        rn = King((self.dimensions - 1, self.dimensions - 1), 1)
        pn1 = Pawn((self.dimensions - 2, self.dimensions - 1), 1)
        pn2 = Pawn((self.dimensions - 3, self.dimensions - 1), 1)
        self.pieces = [rb, pb1, pb2, rn, pn1, pn2]
        self.create_board()


class Game:
    def __init__(self, name='', render=False, dimension=5, sleep=0):
        self.sleep = sleep
        self.name = name
        self.screen = None
        self.board = Board(dimension=dimension)
        self.size = 800 // self.board.dimensions
        self.do_render = render
        self.current_player = 0

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))

        # PC PLAYING
        self.current_case = [0, 0]

    def update(self, events=None):
        self.board.update()
        if self.do_render:
            self.render()
            self.check_events(events)

    def render(self):
        self.screen.fill((20, 20, 20))
        self.board.render(self)

        t1 = Text(f"Current Player: {self.current_player}", (5, 450))
        t1.render(self.screen)

        pygame.draw.rect(self.screen, (0, 100, 255),
                         (self.current_case[0] * self.size, self.current_case[1] * self.size, self.size, self.size),
                         3)  # width = 3
        pygame.display.flip()
        time.sleep(self.sleep)

    def reset(self):
        self.board.reset()
        return self.board.state.T

    def execute(self, action):
        """
        Executes a game action based on the current player and the provided action input.
        Validates whether the action pertains to the correct player. If the action does not
        correspond to the current player, it appends an error to the board's error list and
        prints a warning. If the action is valid, it updates the current player and moves
        the corresponding game piece based on the action.

        :param action: The numeric representation of an action to be performed in the game.
                       Represents both the piece to move and the direction of the move.
        :type action: int

        :return: None
        """
        if self.current_player == 0 and action >= 12:
            print(f"Warning, moving a piece of the other player: {action}")
            self.board.errors.append(3)
        elif self.current_player == 1 and action < 12:
            self.board.errors.append(3)
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
        if events is None:
            events = pygame.event.get()
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
                    self.current_case[1] = (self.current_case[1] - 1) % self.board.dimensions
                elif event.key == pygame.K_s:
                    self.current_case[1] = (self.current_case[1] + 1) % self.board.dimensions
                elif event.key == pygame.K_a:
                    self.current_case[0] = (self.current_case[0] - 1) % self.board.dimensions
                elif event.key == pygame.K_d:
                    self.current_case[0] = (self.current_case[0] + 1) % self.board.dimensions

    def step(self, action=1):
        """
        Executes a single step in the environment based on the given action. The function
        handles the logic for updating the current state, calculating rewards, and determining
        whether the game or process has ended.

        :param action: The action to be executed in the current step.
        :type action: int
        :return: A tuple containing the next state (as a transposed state matrix), the reward
            for the action, and a boolean indicating whether the game/process has ended.
        :rtype: tuple
        """
        self.execute(action)
        self.update()
        next_state = self.board.state.T

        if self.board.errors[-1] != 0:
            return next_state, -20, False
        if self.board.ended:
            # print(f"Game ended, winner: {self.board.winner}")

            if self.board.winner == 1:
                self.reset()
                return next_state, -1, True
            else:
                self.reset()
                return next_state, 100, True

        return next_state, 0, False

    def reward(self):
        return 0


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