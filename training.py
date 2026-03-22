from game_env import Environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.logger import configure
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback2(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:
                log_dict = {
                    "episode/reward": info["episode"]["r"],
                    "episode/length": info["episode"]["l"],
                }

                if "reward_components" in info:
                    for k, v in info["reward_components"].items():
                        log_dict[f"reward_components/{k}"] = v

                wandb.log(log_dict, step=self.num_timesteps)

        return True    

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.rolling_window = 100

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                self.episode_rewards.append(ep_reward)

                # Rolling mean
                rolling_mean = None
                if len(self.episode_rewards) >= self.rolling_window:
                    rolling_mean = sum(
                        self.episode_rewards[-self.rolling_window:]
                    ) / self.rolling_window

                log_dict = {
                    "episode/reward": ep_reward,
                    "episode/length": ep_length,
                }

                if rolling_mean is not None:
                    log_dict["episode/reward_rolling_mean_100"] = rolling_mean

                if "reward_components" in info:
                    for k, v in info["reward_components"].items():
                        log_dict[f"reward_components/{k}"] = v

                wandb.log(log_dict, step=self.num_timesteps)

        return True

    def _on_rollout_end(self):
        # Grab SB3 internal training metrics
        logger_dict = self.model.logger.name_to_value

        train_metrics = {}
        for key, value in logger_dict.items():
            if key.startswith("train/"):
                train_metrics[key] = value

        if train_metrics:
            wandb.log(train_metrics, step=self.num_timesteps)


reward_struct = {
    "hit_made": 1.0,
    "move": 0.01,
    "bullets_missed": -0.05,
    # use this as enemy found for now
    "enemy_in_sight_fired": 10.0,
    "kill": 3.0,
    # optional, already implemented but disabled
    # "death": -10.0,
    # "hit_taken": -1.0,
}

env_settings = {
    "map" : "assets/open_small.png",
    "bot_mode" : "stationary" #surival, 
}

def main():

    wandb.init(
            project = "my_rl_shooter",
            name = f"experiment_6",
            config={"map": env_settings["map"],
                    "bot_mode": env_settings["bot_mode"],
                    }
        )
    

    log_dir = "./sb3_logs"

    new_logger = configure(
        log_dir,
        format_strings=["stdout", "tensorboard"]
    )

    env = Environment(
        ["badboy"],
        541,
        400,
        1,
        None,
        reward_struct,
        env_settings,
        (50, 50)
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

    model.set_logger(new_logger)
    callback = WandbCallback()
    model.learn(
        total_timesteps=500_000,
        callback=callback)
    
    # Save the trained model
    model.save("models/ppo_multiinput_1000k_env")
    wandb.finish()

if __name__ == "__main__":
    main()