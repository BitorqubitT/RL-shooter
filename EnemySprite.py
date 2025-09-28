import pygame

class EnemySprite(pygame.sprite.Sprite):
    def __init__(self, enemy):
        super().__init__()
        self.enemy = enemy
        self.image = pygame.Surface((enemy.width, enemy.height))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()

    def update(self):
        # Small offset to aline visual with center of agent
        self.rect.topleft = (self.enemy.x - 10, self.enemy.y - 10)