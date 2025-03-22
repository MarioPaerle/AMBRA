from time import process_time_ns

import pygame
import numpy as np


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
        self.reset()
        self.errors = []


    def render(self, game):
        pass

    def reset(self):
        self.board = np.zeros((4, 4), dtype=object)



if __name__ == '__main__':
    b = Board()
    print(b.board)