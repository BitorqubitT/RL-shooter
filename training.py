import pygame
from game_env import Environment
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback

class RewardStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_ep = {}
        self.episode_stats = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for info, done in zip(infos, dones):
            comps = info.get("reward_components")
            if comps is not None:
                for k, v in comps.items():
                    self.current_ep[k] = self.current_ep.get(k, 0) + v

            if done:
                # Episode ended — log total sums
                self.episode_stats.append(self.current_ep.copy())

                # Log raw totals
                for k, v in self.current_ep.items():
                    self.logger.record(f"reward/{k}", v)

                # Calculate and log fractions of total reward
                total = sum(self.current_ep.values()) + 1e-8
                for k, v in self.current_ep.items():
                    self.logger.record(f"reward_frac/{k}", v / total)

                self.current_ep.clear()

        return True

wandb.init(
    project="my-rl-shooter",
    name="ppo-run-001",
    config={
        "algo": "PPO",
        "timesteps": 200_000,
    }
)

reward_struct = {
    "hit_made": 1.0,
    "move": 0.01,
    "bullets_missed": -0.05,
    "enemy_in_sight_fired": 0.50,
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
        n_steps=1024,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        ent_coef=0.02,
        device="cuda",
        learning_rate=3e-4
    )

    callback = RewardStatsCallback()
    model.learn(
        total_timesteps=200_000,
        callback=[callback, WandbCallback()])

    # Save the trained model
    model.save("models/ppo_multiinput_100k_env")

if __name__ == "__main__":
    main()