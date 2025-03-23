import pygame
from environment import *
import time

pygame.init()
screen = pygame.display.set_mode((800, 800))

GAME = Game()
GAME.screen = screen

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    if not GAME.board.ended:
        GAME.update(events)
    pygame.display.flip()
