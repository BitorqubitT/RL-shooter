import pygame
from utils import distance
from collections import defaultdict
import math

class bullet():

    def __init__(self, x, y, damage: int, angle: float, speed: float):
        self.x = x
        self.y = y
        self.position = (self.x, self.y)
        self.width = 5
        self.height = 5
        self.damage = damage
        self.angle = angle
        self.speed = speed
        # Calculate direction vector from angle
        self.direction = (
            math.cos(self.angle) * self.speed,
            math.sin(self.angle) * self.speed
        )

    def move(self, direction: tuple):
        # If we are allowed to move.
        #self.position = (self.position.x + direction[0], self.position.y + direction[1])
        self.x += self.direction[0]
        self.y += self.direction[1]
        self.position = (self.x, self.y)

    def destroy():
        return

class bullet_manager():
    def __init__(self, all_players, world_width, world_height):
        self.all_bullets = self._init_bullets(all_players)
        self.world_width = world_width
        self.world_height = world_height

    def _init_bullets(self, all_players):
        #TODO: change these to numpy array
        bullets_per_player = {}
        for i in all_players:
            bullets_per_player[i] = []
        return bullets_per_player

    def check_collision(self, player_positions, wall_coordinates):
        # Get these from environment or put them in bullet manager on init.
        # Might need to use numpy arrays for optimisation
        COLLISION_RADIUS = 10.0

        hit_counts = defaultdict(int)
        hits_made_counts = defaultdict(int)

        hit_counts = {player.player_name: 0 for player in player_positions}
        hits_made_counts = {player.player_name: 0 for player in player_positions}

        for shooter, bullets in self.all_bullets.items():
            new_bullets = []

            for bullet in bullets:
                # Check within bounds
                if not (0.0 <= bullet.x <= self.world_width and 0.0 <= bullet.y <= self.world_height):
                    continue

                hit = False
                for player in player_positions:
                    if player.player_name == shooter:
                        continue

                    if distance((bullet.x, bullet.y), (player.x, player.y)) < COLLISION_RADIUS:
                        hit_counts[player.player_name] += 1
                        hits_made_counts[shooter] += 1
                        hit = True
                        break
                
                if (int(bullet.x), int(bullet.y)) in wall_coordinates:
                    continue

                if not hit:
                    new_bullets.append(bullet)

            self.all_bullets[shooter] = new_bullets

        return dict(hit_counts), dict(hits_made_counts)

    def add_bullet(self, player_name, new_bullet):
        self.all_bullets[player_name].append(new_bullet)

    def update(self):
        # Give list of enemies and other objects?
        # for bullet in bullet,
        # check for collision
        for key in self.all_bullets.keys():
            for bullet in self.all_bullets[key]:
                bullet.move((4.0, 4.0))