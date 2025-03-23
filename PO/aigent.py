from environment import *
import random as rd
import time
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 800))


done = False
game = Game()
game.screen = screen
for i in range(100):
    game.reset()
    done=False
    while not done:
        state = game.board.state.T
        action = rd.randint(0, 23)
        next_state, reward, done = game.step(action)
        if np.linalg.norm(next_state - state) != 0:
            time.sleep(0.5)

