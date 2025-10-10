import os
import random
import math
from PIL import Image
import pygame

from agent import agent
from bullet import bullet_manager
from EnemySprite import EnemySprite
from BulletSprite import BulletSprite

WORLD_WIDTH = 1280
WORLD_HEIGHT = 1024
TILE_SIZE = 1

BACKGROUND_COLOR = (30, 30, 30)
WALL_COLOR = (200, 200, 200)
BULLET_COLOR = (255, 0, 0)
AIM_COLOR = (255, 255, 0)

MAP_PATH = "assets/map1.png"

class Environment:
    def __init__(self, all_players, render=True):
        self.render = render
        self.player_names = all_players
        self.map = self._load_map(MAP_PATH)
        self.bulletmanager = bullet_manager(all_players)
        self._setup_players()

        if self.render:
            self._init_rendering()
        else:
            self._init_offscreen_surface()

    def _setup_players(self):
        self.all_players = []
        for name in self.player_names:
            x, y = self._random_spawn()
            self.all_players.append(agent(x, y, 100, 0, 3, self.bulletmanager, name, self.map))

    def _init_rendering(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))

        self.player_sprites = pygame.sprite.Group()
        self.player_sprite_lookup = {}
        for player in self.all_players:
            sprite = EnemySprite(player)
            self.player_sprites.add(sprite)
            self.player_sprite_lookup[player.player_name] = sprite

        self.bullet_sprites = pygame.sprite.Group()

    def _init_offscreen_surface(self):
        pygame.init()
        self.offscreen_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))

    def _load_map(self, image_path):
        wall_coords = set()
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = image.getpixel((x, y))
                if (r, g, b) == (0, 0, 0) and a == 255:
                    wall_coords.add((x, y))
        return wall_coords

    def _random_spawn(self):
        while True:
            x = random.randint(1, WORLD_WIDTH)
            y = random.randint(1, WORLD_HEIGHT)
            if (x, y) not in self.map:
                return x, y

    def reset(self):
        self._setup_players()

        if self.render:
            self.player_sprites.empty()
            self.player_sprite_lookup = {}

            for player in self.all_players:
                sprite = EnemySprite(player)
                self.player_sprites.add(sprite)
                self.player_sprite_lookup[player.player_name] = sprite

    def _reset_agent(self, player):
        player.x, player.y = self._random_spawn()
        player.hp = 100
        player.alive = True

    def get_player_positions(self):
        return {player.player_name: (player.x, player.y) for player in self.all_players}

    def step(self, keys):
        old_positions = self.get_player_positions()
        deaths = {}

        for player in self.all_players:
            if not player.alive:
                self._reset_agent(player)

        self.all_players[0].action(keys)

        surface = self.screen if self.render else self.offscreen_surface
        surface.fill(BACKGROUND_COLOR)
        self._render_map(surface)

        hit_counts, hit_made_counts = self.bulletmanager.check_collision(self.all_players, self.map)
        self.bulletmanager.update()

        self._apply_hits(hit_counts, deaths)

        if self.render:
            for sprite in self.player_sprites:
                sprite.update()

        self.bullet_sprites = pygame.sprite.Group()
        for bullets in self.bulletmanager.all_bullets.values():
            for bullet in bullets:
                sprite = BulletSprite(bullet)
                sprite.update()
                self.bullet_sprites.add(sprite)

        if self.render:
            self.player_sprites.draw(self.screen)
            self.bullet_sprites.draw(self.screen)
            self._draw_aim_lines(self.screen)
            pygame.display.flip()

        # Compute rewards
        new_positions = self.get_player_positions()
        move_flags = {
            name: int(old_positions[name] != new_positions.get(name))
            for name in old_positions
        }

        all_rewards = self._calculate_rewards(deaths, hit_counts, hit_made_counts, move_flags)
        
        # capute frame
        
        surface = self.screen if self.render else self.offscreen_surface

        #From screen cut pov?
        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # H x W x C format
        
        frame = self._cut_pov(frame, self.all_players[0].position)
        print(frame.shape)
        
        return all_rewards, frame

    def _cut_pov(self, frame, player_position):
        x, y = int(player_position[0]), int(player_position[1])
        half_width, half_height = 150, 150
        start_x = max(0, x - half_width)
        end_x = min(frame.shape[1], x + half_width)
        start_y = max(0, y - half_height)
        end_y = min(frame.shape[0], y + half_height)
        pov_frame = frame[start_y:end_y, start_x:end_x]
        # need padding
        # Make them the same color as the walls?
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
        death_reward=-10
        hit_taken_reward=-1
        hit_made_reward=10
        move_reward=0.2
        

        total_rewards = {}
        for player in self.all_players:
            name = player.player_name
            reward = 0
            reward += deaths.get(name, 0) * death_reward
            reward += hits_taken.get(name, 0) * hit_taken_reward
            reward += hits_made.get(name, 0) * hit_made_reward
            reward += moved.get(name, 0) * move_reward
            total_rewards[name] = reward
        return total_rewards


    def _draw_aim_lines(self, surface):
        aim_length = 30
        for player in self.all_players:
            end_pos = (
                player.position[0] + math.cos(player.angle_pov) * aim_length,
                player.position[1] + math.sin(player.angle_pov) * aim_length
            )
            pygame.draw.line(surface, AIM_COLOR, player.position, end_pos)

    def _render_map(self, surface):
        for (x, y) in self.map:
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(surface, WALL_COLOR, rect)

    def get_screen_image(self):
        """Returns the current visual state as an image (even if not rendering)."""
        surface = self.screen if self.render else self.offscreen_surface
        return pygame.surfarray.array3d(surface)
