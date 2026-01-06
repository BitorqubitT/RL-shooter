import os
import random
import math
from PIL import Image

from agent import agent
from bullet import bullet_manager

import gymnasium as gym
from gymnasium import spaces

import numpy as np

TILE_SIZE = 1

BACKGROUND_COLOR = (30, 30, 30)
WALL_COLOR = (200, 200, 200)
BULLET_COLOR = (255, 0, 0)
AIM_COLOR = (255, 255, 0)

MAP_PATH = "assets/map1.png"

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
#TODO: Can create a clase with Enum for actions

class Environment(gym.Env):

    def __init__(self, all_players, world_width, world_height, render_mode=None):
        self.render_mode = render_mode
        self.player_names = all_players
        self.map = self._load_map(MAP_PATH)
        # For now the agent can only turn one way to aim.
        self.action_space = spaces.Discrete(7)
        #TODO: OBS space might be too big. Find the standard and would it work in this game?
        #TODO: Can i still add other observations
        self.observation_space = spaces.Dict({"agent": spaces.Box(0, 255, shape=(300, 300, 3), dtype=float)})  # Example shape for image observation
        #TODO: implement this
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

        self.world_width = world_width
        self.world_height = world_height
        self.world_view = np.zeros((world_height, world_width, 3), dtype=np.uint8)
        self.bulletmanager = bullet_manager(all_players, world_width, world_height)
        self.all_players = self._setup_players()


    def _setup_players(self) -> list:
        self.all_players = []
        for name in self.player_names:
            x, y = self._random_spawn()
            self.all_players.append(agent(x, y, 100, 0, 3, self.bulletmanager, name, self.map))
        return self.all_players

    def _load_map(self, image_path) -> set:
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
        frame[:, :] = (255, 255, 255)  # Background color

        for wall in self.map:
            x, y = wall
            frame[y, x] = (0, 0, 0) #black walls

        for player in self.all_players:
            if player.alive:
                x, y = int(player.x), int(player.y)
                frame[y:y+player.height, x:x+player.width] = (0, 255, 0)  # Player color

        for shooter, bullets in self.bulletmanager.all_bullets.items():
            for bullet in bullets:
                x, y = int(bullet.x), int(bullet.y)
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    frame[y:y+3, x:x+3] = (255, 0, 0)  # Bullet color
        self.world_view = frame
 
    def _random_spawn(self):
        while True:
            x = random.randint(1, self.world_width)
            y = random.randint(1, self.world_height)
            if (x, y) not in self.map:
                return x, y

    def reset(self, seed=None) -> tuple:
        self._setup_players()

        obs = {"agent":self._cut_pov(self.all_players[0].position)}
        info = {}
        return obs, info

    def _reset_agent(self, player):
        player.x, player.y = self._random_spawn()
        player.hp = 100
        player.alive = True

    def get_player_positions(self):
        return {player.player_name: (player.x, player.y) for player in self.all_players}

    def step(self, keys) -> tuple[dict, float, bool, bool, dict]:
        """
        Args:
            action ( ActType): 

        Returns:
            observation ( ObsType): _description_
            reward (SupportsFloat): _description_
            terminated (bool): _description_
            truncated (bool): _description_
            info (dict):  
        """

        #Remember that we use "real" physics and then we translate to grid world
        #At some point I might need to add non agents.

        #Steps:
        #- Receive a move
        #- Check if anyone died
        #- perform a move (should stay like this I think)
        #- update bullets
        #- check collisions
        #- calculate rewards
        #- return observation, reward, terminated, truncated, info 

        old_positions = self.get_player_positions()
        deaths = {}

        # TODO: Might want to remove resetting of the agent from step function
        for player in self.all_players:
            if not player.alive:
                self._reset_agent(player)

        print("Keys received in step:", keys)
        self.all_players[0].action(keys)

        self.bulletmanager.update()
        hit_counts, hit_made_counts = self.bulletmanager.check_collision(self.all_players, self.map)

        self._apply_hits(hit_counts, deaths)

        # Compute rewards
        new_positions = self.get_player_positions()
        move_flags = {
            name: int(old_positions[name] != new_positions.get(name))
            for name in old_positions
        }


        reward = self._calculate_rewards(deaths, hit_counts, hit_made_counts, move_flags)
        # Change if we ever do multi agent
        reward = reward["badboy"]

        self._update_world_view()
        #observation = {"agent":self._cut_pov(self.all_players[0].position)}
        observation = self.world_view
        # At this point it is not possible for the player to die
        # But there are a lot of other reasons why we would want to reset the env
        # If we dont get any rewards for a long time etc.
        # How does SB3 handle this?
        terminated = False
        #TODO: How to deal with truncated?
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def _cut_pov(self, player_position):
        """
        Fill the padding with:
        a unique “void” tile (often better)
        Always render the agent POV from this padded map
        From the agent’s perspective:
        The world simply ends in walls
        Observation shape never changes
        No need to expose absolute position

        my solution:
        Calculate the extra padding needed based on pov
        first create a big matrix with a certain (3,3,3) value to capture the outside of the map
        Then place the real map in the middle

        #TODO: Pad the matrix
        #padded_matrix = np.full((self.world_height + 300, self.world_width + 300), 0, dtype=np.uint8)

        """

        # Put real frame in the middle

        frame = self.world_view

        #TODO: change this for world view
        #TODO: Add the padding to the world view
        x, y = int(player_position[0]), int(player_position[1])
        half_width, half_height = 150, 150
        start_x = max(0, x - half_width)
        end_x = min(frame.shape[1], x + half_width)
        start_y = max(0, y - half_height)
        end_y = min(frame.shape[0], y + half_height)
        pov_frame = frame[start_y:end_y, start_x:end_x]
        #TODO: add padding -> obs shape
        # Need to add the padding to the correct side
        # We want the agent to learn where the edges are
        # Can also give the arena padding

        return pov_frame

    def _apply_hits(self, hit_counts, deaths):
        # TODO: Change hp amount and move this to bulletmanager?
        for player in self.all_players:
            if player.player_name in hit_counts:
                player.hp -= hit_counts[player.player_name] * 10
                if player.hp <= 0:
                    player.alive = False
                    deaths[player.player_name] = 1
                else:
                    deaths[player.player_name] = 0

    def _calculate_rewards(self, deaths, hits_taken, hits_made, moved):
        death_reward=-10.0
        hit_taken_reward=-1.0
        hit_made_reward=10.0
        move_reward=0.2

        total_rewards = {}
        for player in self.all_players:
            name = player.player_name
            reward = 0.0
            reward += deaths.get(name, 0) * death_reward
            reward += hits_taken.get(name, 0) * hit_taken_reward
            reward += hits_made.get(name, 0) * hit_made_reward
            reward += moved.get(name, 0) * move_reward
            total_rewards[name] = reward
        return total_rewards

    def close(self):
        return None