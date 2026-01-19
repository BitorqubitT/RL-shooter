import pygame
from game_env import Environment
from env_wrapper import EnvWrapper
from stable_baselines3 import PPO

#MAP_PATH = "assets/map1.png"
MAP_PATH = "assets/open.png"

clock = pygame.time.Clock()
running = True
RENDER = None
env = Environment(["badboy"],
                  1280,
                  1024,
                  6,
                  RENDER,
                  "basic", 
                  MAP_PATH)

model = PPO.load("models/ppo_multiinput_100k_env", device="cuda")
pygame.init()
display = pygame.display.set_mode((1280, 1024))

obs, _ = env.reset()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = model.predict(obs, deterministic=False)
    print("Keys:", keys)
    obs, reward, cap1, cap2, cap3 = env.step(keys[0])
    obs_screen = env.world_view
    surf = pygame.surfarray.make_surface(obs_screen.swapaxes(0, 1))
    display.blit(surf, (0, 0))
    pygame.display.flip()
