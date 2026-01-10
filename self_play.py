import pygame
from game_env import Environment
from env_wrapper import EnvWrapper

clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = True

env = Environment(["badboy", "goodboy", "goodestboy", "badboy1", "goodestboy3", "feeliumboy"], 1280, 1024, RENDER)

key_mapping = {
    pygame.K_UP:    0,
    pygame.K_DOWN:  1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT:  3,
    pygame.K_s:     4,
    pygame.K_q:     5,
}

wrapper = EnvWrapper(env, key_mapping)

pygame.init()
display = pygame.display.set_mode((1280, 1024))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    obs, reward, cap1, cap2, cap3 = wrapper.step_from_pygame_keys(keys)

    # Convert obs -> surf, display, etc.
    surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
    display.blit(surf, (0, 0))
    pygame.display.flip()
