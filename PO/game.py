import pygame
from PO.environment import *
import time

pygame.init()
screen = pygame.display.set_mode((800, 800))

GAME = Game(render=True)
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
