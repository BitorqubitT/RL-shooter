import pygame
from game_env import Environment
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

MAP_PATH = "assets/open.png"

# Custom callback to print episodic returns during training
class EpisodicReturnCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'episode' in info:
                print(f"Episode return: {info['episode']['r']}")
        return True

def main():
    env = Environment(
        ["badboy"],
        1280,
        1024,
        6,
        None,
        "basic",
        MAP_PATH
    )

    # Wrap with Monitor to track episode stats
    env = Monitor(env)

    #stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)
    env.reset()

    #TODO: Need to implement steps tracking in my env.


    model = PPO("MultiInputPolicy", env, device="cuda", verbose=1)

    callback = EpisodicReturnCallback()
    model.learn(total_timesteps=100_000, callback=callback)

    # Save the trained model
    model.save("models/ppo_multiinput_env")

if __name__ == "__main__":
    main()