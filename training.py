import pygame
from PIL import Image
import os
from game_env import Environment
import gymnasium as gym
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker


clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = "human"
#RENDER = None
env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feelsboy"], 1280, 1024, RENDER)

stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)
# Takes in ActType( int corresponding to action)
x = env.step(4)
#print(x)




env.reset()
env.render()
env.close()



#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10_000)

# https://gymnasium.farama.org/api/env/#gymnasium.Env.step
# Make it a Gym environment
# Check if env uses the same function names + return types







"""


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