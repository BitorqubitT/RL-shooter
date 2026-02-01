import random
from typing import Dict, List, Tuple, Any

from PIL import Image

from agent import agent
from bullet import bullet_manager

import gymnasium as gym
from gymnasium import spaces

import numpy as np

TILE_SIZE = 1

BACKGROUND_COLOR = (255, 255, 255)
WALL_COLOR = (0, 0, 0)
PLAYER_COLOR = (255, 0, 0)
BOT_COLOR = (0, 0, 255)
BULLET_COLOR = (0, 255, 0)
AIM_COLOR = (255, 255, 0)

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py

class Environment(gym.Env):

    def __init__(
        self,
        all_players: List[str],
        world_width: int,
        world_height: int,
        bot_amount: int,
        render_mode: str,
        reward_struct: Dict[str, float],
        env_settings: Dict[str, Any],
        pov_size: Tuple[int, int] = (150, 150)
    ):
        self.render_mode = render_mode
        self.player_names = all_players
        self.map = self._load_map(env_settings["map"])
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                0,
                255,
                shape=(pov_size[0], pov_size[1], 3),
                dtype=np.uint8
            )
        })
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 60
        }
        self.world_width = world_width
        self.world_height = world_height
        self.pov_size = pov_size
        self.world_view = np.zeros((world_height, world_width, 3), dtype=np.uint8)
        self.bulletmanager = bullet_manager(all_players, world_width, world_height)
        self.all_players = self._setup_players()
        self.all_bots = self._setup_bots(bot_amount)
        self.bot_amount = bot_amount
        self.step_count = 0
        self.explored = set()
        self.reward_struct = reward_struct
        self.env_settings = env_settings


    def _setup_players(self) -> List[agent]:
        players = []
        for name in self.player_names:
            x, y = self._random_spawn()
            players.append(agent(x, y, 100, 0, 3, self.bulletmanager, name, self.map))
        return players
    
    def _setup_bots(self, bot_amount: int) -> List[agent]:
        bots = []
        for i in range(bot_amount):
            x, y = self._random_spawn()
            bots.append(agent(x, y, 100, 0, 3, self.bulletmanager, f"bot_{i}", self.map))
        return bots

    def _load_map(self, image_path: str) -> set[Tuple[int, int]]:
        wall_coords = set()
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = image.getpixel((x, y))
                if (r, g, b) == (0, 0, 0) and a == 255:
                    wall_coords.add((x, y))
        return wall_coords

    def _update_world_view(self) -> np.ndarray:
        frame = np.zeros((self.world_height, self.world_width, 3), dtype=np.uint8)
        frame[:, :] = BACKGROUND_COLOR

        for wall in self.map:
            x, y = wall
            frame[y, x] = WALL_COLOR

        for player in self.all_players:
            if player.alive:
                x, y = int(player.x), int(player.y)
                frame[y:y + player.height, x:x + player.width] = PLAYER_COLOR

        for bot in self.all_bots:
            if bot.alive:
                x, y = int(bot.x), int(bot.y)
                frame[y:y + bot.height, x:x + bot.width] = BOT_COLOR
        
        for bullets in self.bulletmanager.all_bullets.values():
            for bullet in bullets:
                x, y = int(bullet.x), int(bullet.y)
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    frame[y:y + bullet.height, x:x + bullet.width] = BULLET_COLOR
        self.world_view = frame
 
    def _random_spawn(self) -> Tuple[int, int]:
        while True:
            x = random.randint(1, self.world_width - 1)
            y = random.randint(1, self.world_height - 1)
            if (x, y) not in self.map:
                return x, y

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        self.bulletmanager = bullet_manager(self.player_names, self.world_width, self.world_height)
        self.all_players = self._setup_players()
        self.all_bots = self._setup_bots(self.bot_amount)
        self._reset_agent(self.all_players[0])
        self.explored = set()
        self.step_count = 0
        obs = {"agent": self._cut_pov(self.all_players[0])}
        info = {}
        return obs, info

    def _reset_agent(self, player: agent) -> None:
        player.x, player.y = self._random_spawn()
        player.hp = 100
        player.alive = True

    def get_player_positions(self) -> Dict[str, Tuple[float, float]]:
        return {player.player_name: (player.x, player.y) for player in self.all_players}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Args:
            action (int): Action to take

        Returns:
            observation (Dict[str, np.ndarray]): Agent's POV
            reward (float): Reward for the action
            terminated (bool): Whether episode ended due to termination
            truncated (bool): Whether episode ended due to truncation
            info (Dict[str, Any]): Additional info
        """
        deaths = {}
        units_and_players = self.all_players + self.all_bots
        
        for bot in self.all_bots:
            if not bot.alive:
                self._reset_agent(bot)
        
        self.all_players[0].action(action)

        x, y = int(self.all_players[0].position[0]), int(self.all_players[0].position[1])
    
        if (x, y) not in self.explored:
            self.explored.add((x, y))
            move_flags = {self.all_players[0].player_name: 1.0}
        else:
            move_flags = {self.all_players[0].player_name: 0.0}

        self.bulletmanager.update()
        
        hit_counts, hit_made_counts, bullets_missed = self.bulletmanager.check_collision(units_and_players, self.map)

        bots_killed = self._apply_hits(hit_counts)
        if bots_killed > 0:
            bots_killed = 1

        terminated = False
        if hit_made_counts["badboy"] > 0:
            terminated = True 

        enemy_in_sight_fired = {} 
        if self._enemy_in_sight(self.all_players[0]) and action == 4:
            enemy_in_sight_fired["badboy"] = 1.0

        reward = self._calculate_rewards(deaths, 
                                         hit_counts, 
                                         hit_made_counts, 
                                         move_flags, 
                                         bullets_missed, 
                                         enemy_in_sight_fired,
                                         bots_killed)
        reward = reward["badboy"]

        self._update_world_view()
        observation = {"agent": self._cut_pov(self.all_players[0])}
        truncated = False
        info = {}
        self.step_count += 1

        if self.step_count == 1000:
            truncated = True

        info["reward_components"] = {
            "hit_made": hit_made_counts.get("badboy", 0),
            "move": move_flags.get("badboy", 0),
            "bullets_missed": bullets_missed.get("badboy", 0),
            "enemy_in_sight_fired": enemy_in_sight_fired.get("badboy", 0),
            "kill": bots_killed
        }

        return observation, reward, terminated, truncated, info

    def _cut_pov(self, player: agent) -> np.ndarray:
        """Create a point of view of the player. Add padding for a consistent pov shape.

        Args:
            player: Player object

        Returns:
            pov_frame: Cut out POV
        """
        pov_half_height = int(self.pov_size[0] / 2)
        pov_half_width = int(self.pov_size[1] / 2)

        frame = self.world_view

        px = int(player.position[0])
        py = int(player.position[1])

        padded = np.pad(
            frame,
            pad_width=(
                (pov_half_height, pov_half_height),
                (pov_half_width, pov_half_width),
                (0, 0)
            ),
            mode="constant",
            constant_values=0
        )

        py += pov_half_height
        px += pov_half_width

        pov_frame = padded[
            py - pov_half_height : py + pov_half_height,
            px - pov_half_width  : px + pov_half_width
        ] 
        
        return pov_frame
    
    def _enemy_in_sight(self, player: agent) -> bool:
        px, py = int(player.x), int(player.y)
        for other in self.all_players + self.all_bots:
            if other.player_name != player.player_name and other.alive:
                ox, oy = int(other.x), int(other.y)
                if abs(px - ox) <= 85 and abs(py - oy) <= 85:
                    return True
        return False

    def _apply_hits(self, hit_counts: Dict[str, int]) -> int:
        deaths = {}
        all_units = self.all_players + self.all_bots
        for player in all_units:
            if player.player_name in hit_counts:
                player.hp -= hit_counts[player.player_name] * 10
                if player.hp <= 0:
                    player.alive = False
                    deaths[player.player_name] = 1
                else:
                    deaths[player.player_name] = 0
        return sum(deaths.values())

    def _calculate_rewards(
        self,
        deaths: Dict[str, int],
        hits_taken: Dict[str, int],
        hits_made: Dict[str, int],
        moved: Dict[str, float],
        bullets_missed: Dict[str, int],
        enemy_in_sight_fired: Dict[str, float],
        bots_killed: int
    ) -> Dict[str, float]:

        rs = self.reward_struct

        total_rewards = {}

        for player in self.all_players:
            name = player.player_name
            reward = 0.0

            if "death" in rs:
                reward += deaths.get(name, 0) * rs["death"]

            if "hit_taken" in rs:
                reward += hits_taken.get(name, 0) * rs["hit_taken"]

            if "hit_made" in rs:
                reward += hits_made.get(name, 0) * rs["hit_made"]

            if "move" in rs:
                reward += moved.get(name, 0) * rs["move"]

            if "bullets_missed" in rs:
                reward += bullets_missed.get(name, 0) * rs["bullets_missed"]

            if "enemy_in_sight_fired" in rs:
                reward += enemy_in_sight_fired.get(name, 0) * rs["enemy_in_sight_fired"]

            if "kill" in rs:
                reward += bots_killed * rs["kill"]

            total_rewards[name] = reward

        return total_rewards

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.world_view
        return None

    def close(self) -> None:
        # No resources to clean up
        pass