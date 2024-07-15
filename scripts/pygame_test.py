import pygame
import sys
import os

# Initialize Pygame
os.environ["SDL_VIDEODRIVER"] = "x11"
pygame.init()
try:
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Pygame Basic Test')
    print("Pygame display initialized.")
except pygame.error as e:
    print(f"Error initializing Pygame display: {e}")
    sys.exit()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 0, 0))  # Fill the screen with red
    pygame.display.flip()
