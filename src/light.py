from __future__ import annotations

import random
from functools import cached_property
from typing import Generator

from scene import Scene
from vector3 import Vector3, cross


def _build_orthogonal_basis(direction: Vector3, radius: float):
    if abs(direction.x) > abs(direction.z):
        tangent = Vector3(-direction.y, direction.x, 0)
    else:
        tangent = Vector3(0, -direction.z, direction.y)

    tangent = tangent.normalized
    bitangent = cross(direction, tangent).normalized

    return tangent * radius, bitangent * radius


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @cached_property
    def get_position(self) -> Vector3:
        return Vector3.from_array(self.position)

    def samples(self, direction: Vector3) -> Generator[Vector3]:
        t, b = _build_orthogonal_basis(direction.normalized, self.radius)
        top_left = self.get_position - (t + b) / 2

        n = Scene().settings.root_number_shadow_rays

        delta_t = t / n
        delta_b = b / n

        for x in range(int(n)):
            for y in range(int(n)):
                yield top_left + delta_t * (x + random.random()) + delta_b * (y + random.random())

