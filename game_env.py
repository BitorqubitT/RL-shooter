import random
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

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
#TODO: Can create a clase with Enum for actions

class Environment(gym.Env):

    def __init__(self, all_players, world_width, world_height, bot_amount, render_mode, mode, map_name, pov_size=(150, 150)):
        self.render_mode = render_mode
        self.player_names = all_players
        self.map = self._load_map(map_name)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                0, 
                255, 
                shape=(pov_size[0], pov_size[1], 3), 
                dtype=np.uint8)
        })
        #TODO: REMOVE?
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
              "render_fps": 60}
        self.mode = mode
        self.world_width = world_width
        self.world_height = world_height
        self.pov_size = pov_size
        self.world_view = np.zeros((world_height, world_width, 3), dtype=np.uint8)
        self.bulletmanager = bullet_manager(all_players, world_width, world_height)
        self.all_players = self._setup_players()
        # Bots are non agent units.
        self.all_bots = self._setup_bots(bot_amount)
        self.bot_amount = bot_amount
        self.step_count = 0
        self.did_we_kill = False
        #TODO: naive implementation maybe put this in agent later.
        self.explored = set()


    def _setup_players(self) -> list:
        self.all_players = []
        for name in self.player_names:
            x, y = self._random_spawn()
            self.all_players.append(agent(x, y, 100, 0, 3, self.bulletmanager, name, self.map))
        return self.all_players
    
    def _setup_bots(self, bot_amount) -> list:
        self.all_bots = []
        for i in range(0, bot_amount):
            x, y = self._random_spawn()
            self.all_bots.append(agent(x, y, 100, 0, 3, self.bulletmanager, f"bot_{i}", self.map))
        return self.all_bots

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
                frame[y:y+player.height, x:x+player.width] = (255, 0, 0)

        # BOTS
        for bot in self.all_bots:
            if bot.alive:
                x, y = int(bot.x), int(bot.y)
                frame[y:y+bot.height, x:x+bot.width] = (0, 0, 255)
        
        for shooter, bullets in self.bulletmanager.all_bullets.items():
            for bullet in bullets:
                x, y = int(bullet.x), int(bullet.y)
                if 0 <= x < self.world_width and 0 <= y < self.world_height:
                    frame[y:y+bullet.height, x:x+bullet.width] = (0, 255, 0)  # Bullet color
        self.world_view = frame
 
    def _random_spawn(self):
        while True:
            x = random.randint(1, self.world_width)
            y = random.randint(1, self.world_height)
            if (x, y) not in self.map:
                return x, y

    def reset(self, seed=None) -> tuple:
        self.bulletmanager = bullet_manager(self.player_names, self.world_width, self.world_height)
        self._setup_players()
        self._setup_bots(self.bot_amount)
        self._reset_agent(self.all_players[0])
        self.explored = set()
        self.step_count = 0
        obs = {"agent":self._cut_pov(self.all_players[0])}
        #obs = self.world_view
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
        deaths = {}
        units_and_players = self.all_players + self.all_bots
        
        # TODO: Might want to remove resetting of the agent from step function
        for player in units_and_players:
            if not player.alive:
                self._reset_agent(player)
        self.all_players[0].action(keys)


        x, y = int(self.all_players[0].position[0]), int(self.all_players[0].position[1])
    
        if (x, y) not in self.explored:
            self.explored.add((x, y))
            move_flags = {self.all_players[0].player_name: 1.0}
        else:
            move_flags = {self.all_players[0].player_name: 0.0}

        self.bulletmanager.update()
        
        hit_counts, hit_made_counts, bullets_missed = self.bulletmanager.check_collision(units_and_players, self.map)

        #TODO: Is this going well with deaths? Check values
        #TODO: Fix this later
        bots_killed = self._apply_hits(hit_counts)
        if bots_killed > 0:
            print("Bots killed:", bots_killed)
            bots_killed = 1

        terminated = False
        #might make trianing worse
        if hit_made_counts["badboy"] > 0:
            terminated = True 


        enemy_in_sight_fired = {} 
        if self._enemy_in_sight(self.all_players[0]):
            enemy_in_sight_fired["badboy"] = 5.0

        reward = self._calculate_rewards(deaths, 
                                         hit_counts, 
                                         hit_made_counts, 
                                         move_flags, 
                                         bullets_missed, 
                                         enemy_in_sight_fired,
                                         bots_killed)
        # Change if we ever do multi agent
        reward = reward["badboy"]

        self._update_world_view()
        observation = {"agent":self._cut_pov(self.all_players[0])}
        #observation = self.world_view
        truncated = False
        info = {}
        self.step_count += 1

        #TODO: player cant die yet

        #TODO: Only do this for training
        if self.step_count == 1000:
            truncated = True

        return observation, reward, terminated, truncated, info

    def _cut_pov(self, player) -> np.ndarray:
        """Create a point of view of the player. Add padding for a consistent pov shape.

        Args:
            player (_type_): Player object
            pov_half_width (int, optional): Half the pov size that we want. Defaults to 150.
            pov_half_height (int, optional): Half the pov size that we want. Defaults to 150.

        Returns:
            pov_frame: Cut out pov
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
    
    def _enemy_in_sight(self, player) -> bool:
        px, py = int(player.x), int(player.y)
        for other in self.all_players + self.all_bots:
            if other.player_name != player.player_name and other.alive:
                ox, oy = int(other.x), int(other.y)
                #TODO: Look at other options
                if abs(px - ox) <= 150 and abs(py - oy) <= 150:
                    return True
        return False

    def _apply_hits(self, hit_counts):
        deaths = {}
        #TODO: How to deal with bots and agents
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

    def _calculate_rewards(self, 
                           deaths, 
                           hits_taken, 
                           hits_made, 
                           moved, 
                           bullets_missed, 
                           enemy_in_sight_fired,
                           bots_killed) -> dict:
        #kill reward
        #death_reward=-10.0
        #hit_taken_reward=-1.0
        hit_made_reward=20.0
        move_reward=0.1
        bullets_missed_reward=-0.05
        fired_while_enemy_in_sight_reward=1.0
        kill_made = 30.0

        #TODO: Can we add a kill reward?
        total_rewards = {}
        for player in self.all_players:
            name = player.player_name
            reward = 0.0
            #reward += deaths.get(name, 0) * death_reward
            #reward += hits_taken.get(name, 0) * hit_taken_reward
            reward += hits_made.get(name, 0) * hit_made_reward
            reward += moved.get(name, 0) * move_reward
            reward += bullets_missed.get(name, 0) * bullets_missed_reward
            #TODO: CAn be positive or negative atm. VAGUE
            reward += enemy_in_sight_fired.get(name, 0) * fired_while_enemy_in_sight_reward
            reward += bots_killed * kill_made
            total_rewards[name] = reward
        return total_rewards

    def close(self):
        return None