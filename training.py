from game_env import Environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from stable_baselines3.common.logger import configure
import wandb
from stable_baselines3.common.callbacks import BaseCallback

import hydra
from omegaconf import OmegaConf

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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):

    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    env = Environment(
        cfg.env.all_players,
        cfg.env.world_size[0],
        cfg.env.world_size[1],
        cfg.env.num_bots,
        None,
        cfg.reward,
        cfg.env,
        cfg.env.pov, # sloppy, get this from prev config
    )

    env = Monitor(env)
    env.reset()
    
    model = RecurrentPPO(
        MultiInputLstmPolicy,
        env,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        ent_coef=cfg.train.ent_coef,
        learning_rate=cfg.train.learning_rate,
        device="cuda",
        verbose=1,
    )

    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=WandbCallback()
    )

    wandb.finish()

    # use config combo + result for the name
    model.save("models/" + cfg.experiment.name)

if __name__ == "__main__":
    main()