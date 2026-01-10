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

while running:


    keys = 4

    obs, reward, cap1, cap2, cap3 = env.step(keys)

    # Save frame as image
    #frame_image = Image.fromarray(obs['agent'], 'RGB')
    frame_image = Image.fromarray(obs)
    frame_image.save(f"frames_per_run/frame_{i}.png")

    clock.tick(FPS)
    i += 1

