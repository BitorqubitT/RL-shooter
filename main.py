import pygame
from PIL import Image
import os
from game_env import Environment
import numpy as np

clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = True
env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feelsboy"], 1280, 1024, RENDER)
i = 0

pygame.init()
#display = pygame.display.set_mode((1280, 1024))
display = pygame.display.set_mode((1280, 1024))
#obs = np.zeros((1024, 1280, 3), dtype=np.uint8)
#surf = pygame.surfarray.make_surface(obs)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #keys = pygame.key.get_pressed()
    keys = 4
    obs, reward, cap1, cap2, cap3 = env.step(keys)
    print(reward)



    surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))

    display.blit(surf, (0, 0))
    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()

    # Save frame as image
    #frame_image = Image.fromarray(obs['agent'], 'RGB')
    #frame_image = Image.fromarray(obs)
    #frame_image.save(f"frames_per_run/frame_{i}.png")

