import pygame
from game_env import Environment
from env_wrapper import EnvWrapper
from stable_baselines3 import PPO

clock = pygame.time.Clock()
running = True
RENDER = None

reward_struct = {
    "hit_made": 1.0,
    "move": 0.001,
    "bullets_missed": -0.05,
    "enemy_in_sight_fired": 0.1,
    "kill": 3.0,
    # optional, already implemented but disabled
    # "death": -10.0,
    # "hit_taken": -1.0,
}

env_settings = {
    "map" : "assets/open_small.png",
    "bot_mode" : "stationary" #surival, 
}

env = Environment(
    ["badboy"],
    541,
    400,
    1,
    RENDER,
    reward_struct, 
    env_settings,
    (50, 50)
    )

model = PPO.load("models/ppo_multiinput_1000k_env", device="cuda")
pygame.init()
display = pygame.display.set_mode((541, 400))

obs, _ = env.reset()

total_reward = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = model.predict(obs, deterministic=False)
    print(type(keys))
    obs, reward, cap1, cap2, cap3 = env.step(keys[0])
    print(cap3)
    total_reward += reward
    #TODO: would be nice to see distribution of where the reward is coming from
    print(f"Total reward: {total_reward}")
    #obs_screen = env.world_view
    obs2 = obs["agent"]
    surf = pygame.surfarray.make_surface(obs2.swapaxes(0, 1))
    display.blit(surf, (0, 0))
    pygame.display.flip()