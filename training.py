import pygame
from game_env import Environment
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

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
        25,
        None,
        "basic",
        MAP_PATH
    )

    # Wrap with Monitor to track episode stats
    env = Monitor(env)

    #stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)
    env.reset()

    #TODO: Need to implement steps tracking in my env.


    #model = PPO("MultiInputPolicy", env, device="cuda", verbose=1)
    
    model = RecurrentPPO(
        MultiInputLstmPolicy,
        env,
        n_steps=128,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        ent_coef=0.01,
        device="cuda",
        learning_rate=5e-5
    )

    callback = EpisodicReturnCallback()
    model.learn(total_timesteps=100_000, callback=callback)

    # Save the trained model
    model.save("models/ppo_multiinput_100k_env")

if __name__ == "__main__":
    main()