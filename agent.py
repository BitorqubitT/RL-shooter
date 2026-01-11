from bullet import bullet

class agent():

    def __init__(self, x, y, hp: int, angle_pov: float, speed: float, bullet_manager, player_name, map):
        self.player_name = player_name
        self.x = x
        self.y = y
        self.width = 20
        self.height = 20
        self.position = (self.x, self.y)
        self.hp = hp
        self.angle_pov = angle_pov
        self.speed = speed
        self.bullet_manager = bullet_manager
        self.alive = True
        self.map = map

    def move(self, direction: tuple):
        #TODO: Come up with a nicer solution
        new_x = (self.x + (direction[0] * self.speed))
        new_y = (self.y + (direction[1] * self.speed))
        if (new_x, new_y) in self.map or new_x < 0 or new_x > 1280 or new_y < 0 or new_y > 1024:
            return
        else:
            self.x += direction[0] * self.speed
            self.y += direction[1] * self.speed
            self.position = (self.x + 0.1, self.y + 0.1)

    def shoot(self):
        new_bullet = bullet(self.x, self.y, 5, self.angle_pov, 15)
        self.bullet_manager.add_bullet(self.player_name, new_bullet)

    def die():
        return

    def action(self, keys):
        if isinstance(keys, int):
            mapping = {
                0: lambda: self.move((0, -1)),
                1: lambda: self.move((0, 1)),
                2: lambda: self.move((1, 0)),
                3: lambda: self.move((-1, 0)),
                4: self.shoot,
                5: lambda: setattr(self, "angle_pov", self.angle_pov - 0.1),
                #6: lambda: setattr(self, "angle_pov", self.angle_pov + 0.1),
            }
            action = mapping.get(keys)
            if action:
                action()
