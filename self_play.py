import pygame
from game_env import Environment
from env_wrapper import EnvWrapper
from pygame.locals import QUIT

pygame.init()
display_width, display_height = 541, 400
display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("RL Shooter - Human Play")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

reward_struct = {
    "hit_made": 1.0,
    "move": 0.01,
    "bullets_missed": -0.05,
    "enemy_in_sight_fired": 0.5,
    "kill": 3.0,
}

env_settings = {
    "map": "assets/open_small.png",
    "bot_mode": "stationary"
}

env = Environment(
    ["badboy"],
    display_width,
    display_height,
    1,
    None,  # RENDER not needed
    reward_struct,
    env_settings,
    (50, 50)
)

key_mapping = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_s: 4,
    pygame.K_q: 5,
}

wrapper = EnvWrapper(env, key_mapping, default_action=None)

obs = wrapper.reset()
total_reward = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    keys = pygame.key.get_pressed()
    result = wrapper.step_from_pygame_keys(keys)

    if result is not None:
        obs, reward, cap1, cap2, cap3 = result
        total_reward += reward
        print(f"Reward: {reward:.3f}, Total: {total_reward:.2f}")

    obs_img = obs["agent"]
    surf = pygame.surfarray.make_surface(obs_img.swapaxes(0, 1))
    display.blit(surf, (0, 0))

    reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
    display.blit(reward_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)