import pygame
from game_env import Environment

clock = pygame.time.Clock()
FPS = 60
running = True
RENDER = True
env = Environment(["badboy", "goodboy", "goodestboy"], RENDER)
i = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    x = env.step(keys)
    print(x)
    clock.tick(FPS)
pygame.quit()

"""
To think:
The game doesnt really need rounds or whatever. But giving the agent a complete new change etc might be important
Create a way of customising all the rewards with a matrix into the environment.


TODO:
- Add state and thus point of view
    - Raw pixel view
    - Can later add vector with pos etc.

- Make sure everything works for multiple agents 

- add replay system.
- give every agent a color
- add image based training


These versions are all for training.
- version one
    - kill standing still agents
    
- version two.
    - Make agents move randomly (path tracing)

Do I need to start making optimisations?
Matrix operations only?

    

- Version 4:
    - teambased


"""
