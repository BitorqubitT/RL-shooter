import pygame
from PIL import Image
import os
from game_env import Environment
import gymnasium as gym
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker


clock = pygame.time.Clock()
running = True
RENDER = None
#RENDER = None
env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feelsboy"], 1280, 1024, RENDER)

stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

env.reset()

model = PPO("MultiInputPolicy", env, device="cuda", verbose=1)
model.learn(total_timesteps=10_000)

model.save("models/ppo_multiinput_env")
