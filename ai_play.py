import pygame
from PIL import Image
from game_env import Environment

MAP_PATH = "assets/map1.png"
clock = pygame.time.Clock()
FPS = 30
running = True
RENDER = True
env = Environment(["badboy"],
                   1280,
                   1024,
                   6,
                   RENDER,
                   "basic",
                   MAP_PATH)
i = 0

while running:


    keys = 4

    obs, reward, cap1, cap2, cap3 = env.step(keys)

    # Save frame as image
    frame_image = Image.fromarray(obs['agent'], 'RGB')
    #frame_image = Image.fromarray(obs)
    frame_image.save(f"frames_per_run/frame_{i}.png")

    clock.tick(FPS)
    i += 1

