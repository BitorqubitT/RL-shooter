import pygame
from game_env import Environment
from env_wrapper import EnvWrapper

#MAP_PATH = "assets/map1.png"
MAP_PATH = "assets/open.png"

clock = pygame.time.Clock()
running = True
RENDER = None

reward_struct = {
    "hit_made": 1.0,
    "move": 0.01,
    "bullets_missed": -0.05,
    "enemy_in_sight_fired": 0.5,
    "kill": 3.0,
    # optional, already implemented but disabled
    # "death": -10.0,
    # "hit_taken": -1.0,
}

env_settings = {
    "map" : "assets/open.png",
    "bot_mode" : "stationary" #surival, 
}

env = Environment(["badboy"],
                  1280,
                  1024,
                  25,
                  RENDER,
                  reward_struct, 
                  env_settings,
                  (150, 150)
                  )

key_mapping = {
    pygame.K_UP:    0,
    pygame.K_DOWN:  1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT:  3,
    pygame.K_s:     4,
    pygame.K_q:     5,
}

wrapper = EnvWrapper(env, key_mapping)

pygame.init()
display = pygame.display.set_mode((1280, 1024))

total_reward = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    obs, reward, cap1, cap2, cap3 = wrapper.step_from_pygame_keys(keys)
    print(cap3)
    total_reward += reward
    print(f"Total reward: {total_reward}")
    obs = obs["agent"]
    # Convert obs -> surf, display, etc.
    surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
    display.blit(surf, (0, 0))
    pygame.display.flip()
