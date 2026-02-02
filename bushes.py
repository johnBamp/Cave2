from typing import Tuple

import pygame

import settings

Vec2 = pygame.math.Vector2


class Bush:
    def __init__(self, cell_x: int, cell_y: int, cell_size: int):
        self.cell = (cell_x, cell_y)
        self.cell_size = cell_size
        self.center = Vec2(cell_x * cell_size + cell_size / 2, cell_y * cell_size + cell_size / 2)
        self.has_fruit = True
        self._regen_timer = 0.0

    def update(self, dt: float):
        if self.has_fruit:
            return
        self._regen_timer += dt
        if self._regen_timer >= settings.FRUIT_REGEN_SECONDS:
            self.has_fruit = True
            self._regen_timer = 0.0

    def draw(self, surface: pygame.Surface):
        radius = self.cell_size * 0.45
        pygame.draw.circle(surface, settings.BUSH_COLOR, self.center, radius)
        if self.has_fruit:
            pygame.draw.circle(surface, settings.FRUIT_COLOR, self.center, radius * 0.45)

    def contains_point(self, pos: Vec2) -> bool:
        # Treat bush as its tile; quick AABB check.
        half = self.cell_size / 2
        if abs(pos.x - self.center.x) > half or abs(pos.y - self.center.y) > half:
            return False
        return True

    def consume(self):
        self.has_fruit = False
        self._regen_timer = 0.0
