from turtle import width
from agent import agent
import pygame
from bullet import bullet_manager
from EnemySprite import EnemySprite
from BulletSprite import BulletSprite
from PIL import Image
import random
import math

WORLD_WIDTH = 1280
WORLD_HEIGHT = 1024
TILE_SIZE = 1
BACKGROUND_COLOR = (30, 30, 30)
BULLET_COLOR = (255, 0, 0)

class Environment():

    def __init__(self, all_players, render):
        self.all_players = []
        self.all_sprites = []
        self.bulletmanager = bullet_manager(all_players)
        self.player_names = all_players
        self.map = self._create_map()
        self._setup(all_players)
        self.render = render
        self._init_pygame()
        #self._init_pygame_offscreen()
    
    def _setup(self, all_players):
        # Create user
        # TODO: Give a list with all players
        for i in all_players:
            #TODO: check if this works
            x = random.randint(1, 1280)
            y = random.randint(1, 1024)
            while (x, y) in self.map:
                x = random.randint(1, 1280)
                y = random.randint(1, 1024)
            self.all_players.append(agent(x, y, 100, 0, 3, self.bulletmanager, i, self.map))
       
    def _init_pygame(self):
        #TODO: What do we need to run it without visualising?
        self.screen = pygame.display.set_mode((1280, 1024))
        pygame.display.set_caption("Hello Pygame")
        if self.render:
            self.player_sprite_group = pygame.sprite.Group()
            self.bullet_sprite_group = pygame.sprite.Group()
            self.all_sprites = []
            for player in self.all_players:
                self.all_sprites.append(EnemySprite(player))

    def _init_pygame_offscreen(self):
        world_surface = pygame.Surface((1280, 1024))
        world_surface.fill(BACKGROUND_COLOR)
        for (x, y) in self.map:
            pygame.draw.rect(world_surface, (200, 200, 200), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        for player in self.bulletmanager.all_bullets.keys():
            for bullet in self.bulletmanager.all_bullets[player]:
                pygame.draw.circle(world_surface, BULLET_COLOR, (int(bullet.x), int(bullet.y)), bullet.radius)
        #for player in self.all_players:


    def _create_map(self) -> set:
        # Make more robust and get image location through main.py
        wall_coords = set()
        image =  Image.open("assets/map1.png").convert("RGBA")
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = image.getpixel((x, y))
                # added alpha because images are mostly transparent
                if (r, g, b) == (0, 0, 0) and a == 255:
                    wall_coords.add((x, y))
        return wall_coords

    def get_player_positions(self):
        all_players_positions = {}
        for player in self.all_players:
            all_players_positions[player.player_name] = (player.x, player.y)
        return all_players_positions
    
    def reset(self):
        self.all_players = []
        self._setup(self.player_names)
        if self.render:
            self._init_pygame()
    
    def _reset_agent(self, player):
        x = random.randint(1, 1280)
        y = random.randint(1, 1024)
        while (x, y) in self.map:
            x = random.randint(1, 1280)
            y = random.randint(1, 1024)
        player.x = x
        player.y = y
        player.hp = 100
        player.alive = True 

    def step(self, keys):
        old_pos = self.get_player_positions()

        deaths = {}

        for player in self.all_players:
            if player.alive is False:
                self._reset_agent(player)

        # TODO: I reset position for enemies do i really need to recreate?
        # TODO: Or only for bullets?

        if self.render:
            self.player_sprite_group = pygame.sprite.Group()
            bullet_sprites = pygame.sprite.Group()
            for i in self.all_sprites:
                self.player_sprite_group.add(i)
            # Clear screen
            self.screen.fill((30, 30, 30))  # Dark background

        self.all_players[0].action(keys)
        
        for i in self.all_sprites:
            i.update()
        
        # Load all players into the environment
        # Maybe also check if we want to visualise there.
        #
        #TODO: Check returns to damage players
        hit_counts, hit_made_counts = self.bulletmanager.check_collision(self.all_players, self.map)
        self.bulletmanager.update()

        total_bullets = 0
        for i in self.bulletmanager.all_bullets.keys():
            total_bullets += len(self.bulletmanager.all_bullets[i])

        for player in self.all_players:
            if player.player_name in hit_counts:
                player.hp -= hit_counts[player.player_name] * 10
                if player.hp <= 0:
                    player.alive = False
                    deaths[player.player_name] = 1
                else:
                    deaths[player.player_name] = 0

        # TODO: Only do below if we are visualising
        for player in self.bulletmanager.all_bullets.keys():
            for bullet in self.bulletmanager.all_bullets[player]:
                bullet_sprite = BulletSprite(bullet)
                bullet_sprites.add(bullet_sprite)
                bullet_sprite.update()

        if self.render:
            self.player_sprite_group.draw(self.screen)
            bullet_sprites.draw(self.screen)
            self._show_aim()
            TILE_SIZE = 1
            for (x, y) in self.map:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, (200, 200, 200), rect)

        new_pos = self.get_player_positions()
        
        did_the_player_move = {}
        for player_name, old_position in old_pos.items():
            new_position = new_pos.get(player_name)
            if new_position and new_position != old_position:
                did_the_player_move[player_name] = 1
            else:
                did_the_player_move[player_name] = 0

        total_rewards = self._calculate_rewards(deaths, hit_counts, hit_made_counts, did_the_player_move)
        #return_per_car.append([car.state, reward, hit_wall_check, finished])
        # return a list of states and use slicing in the env to get the right info.
        return total_rewards 
    
    def _calculate_rewards(self, deaths, hit_counts, hit_made_counts, did_the_player_move):
        death_reward = -10
        hit_reward = -1
        hit_made_reward = 10
        move_reward = 0.2

        total_scores = {}
        for player in self.all_players:
            player_score = 0
            player_score += deaths.get(player.player_name) * death_reward
            player_score += hit_counts.get(player.player_name) * hit_reward
            player_score += hit_made_counts.get(player.player_name) * hit_made_reward
            player_score += did_the_player_move.get(player.player_name) * move_reward
            total_scores[player.player_name] = player_score

        return total_scores

    def _show_aim(self):
        aim_length = 30
        for player in self.all_players:
            aim_position = (
                player.position[0] + math.cos(player.angle_pov) * aim_length,
                player.position[1] + math.sin(player.angle_pov) * aim_length
            )
            print(player.x, player.y, player.position)
            pygame.draw.line(self.screen, (255, 255, 0), aim_position, player.position)
        return
