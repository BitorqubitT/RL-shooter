import pygame

class BulletSprite(pygame.sprite.Sprite):
    def __init__(self, bullet):
        super().__init__()
        self.bullet = bullet
        self.image = pygame.Surface((bullet.width, bullet.height))
        self.image.fill((255, 255, 0))  # Yellow bullet
        self.rect = self.image.get_rect()

    def update(self):
        # Sync sprite position to logic bullet
        self.rect.topleft = (self.bullet.x, self.bullet.y)