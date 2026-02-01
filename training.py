import pygame
from game_env import Environment
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

# Custom callback to print episodic returns during training
class EpisodicReturnCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'episode' in info:
                print(f"Episode return: {info['episode']['r']}")
        return True

reward_struct = {
    "hit_made": 1.0,
    "move": 0.01,
    "bullets_missed": -0.05,
    "enemy_in_sight_fired": 0.05,
    "kill": 3.0,
    # optional, already implemented but disabled
    # "death": -10.0,
    # "hit_taken": -1.0,
}

env_settings = {
    "map" : "assets/open.png",
    "bot_mode" : "stationary" #surival, 
}

def main():
    env = Environment(
        ["badboy"],
        1280,
        1024,
        25,
        None,
        reward_struct,
        env_settings,
        (150, 150)
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