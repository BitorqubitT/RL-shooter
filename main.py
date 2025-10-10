import pygame
from PIL import Image
import os
from game_env import Environment

clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = True
env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feelsboy"], RENDER)
i = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    x, frame = env.step(keys)
    # Save frame as image
    frame_image = Image.fromarray(frame)
    frame_image.save(f"frames_per_run/frame_{i}.png")
    i += 1
    clock.tick(FPS)
pygame.quit()

"""
To think:
The game doesnt really need rounds or whatever. But giving the agent a complete new env etc might be important

FIXME:
- Pov when no rendering doenst work yet
- render of aim is bugged (can also remove it for real runs and just add a number for aim)
- can shoot in the whole map

TODO:

- Later can add matrix world or different vectors

- Make sure everything works for multiple agents 

- add replay system.
- give every agent a color
- add image based training


These versions are all for training.
- version one
    - kill standing still agents
    
- version two.
    - Make agents move randomly (path tracing)

Do I need to start making optimisations?
Matrix operations only?

    

- Version 4:
    - teambased


"""
