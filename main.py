import pygame
from PIL import Image
import os
from game_env import Environment

clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = True
env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feelsboy"], 1280, 1024, RENDER)
i = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    #keys = 4
    x, frame = env.step(keys)
    # Save frame as image
    #frame_image = Image.fromarray(frame)
    #frame_image.save(f"frames_per_run/frame_{i}.png")
    i += 1
    clock.tick(FPS)
pygame.quit()

"""
To think:
The game doesnt really need rounds or whatever. But giving the agent a complete new env etc might be important

FIXME:

TODO:
#1
- Create nicer world
- Give every agent a color
- Create first RL version, shoot standint still agents
- Make sure I can replay a game.
    - Not by using images (too static)
    - Get starting position and then all actions

#2
- Make agents move randomly and shoot?(path tracing)


#IDEAS:

- Add matrix world or different vectors
- Make sure everything works for multiple agents (battle at the same time, train)
- add replay system.
Do I need to start making optimisations?
Matrix operations only?
Team based?

"""
