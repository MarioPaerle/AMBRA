import pygame
from environment import *
import time

pygame.init()
screen = pygame.display.set_mode((600, 600))

GAME = Game()
GAME.screen = screen

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    GAME.update()
    pygame.display.flip()
    GAME.board.pieces[0].move(GAME.board, 0)
    time.sleep(0.5)
    print(GAME.board.errors)
