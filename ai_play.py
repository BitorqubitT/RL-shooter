import pygame
from game_env import Environment
from env_wrapper import EnvWrapper
from stable_baselines3 import PPO

import hydra
from omegaconf import OmegaConf


clock = pygame.time.Clock()
running = True
RENDER = None

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):

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

if __name__ == "__main__":
    main()