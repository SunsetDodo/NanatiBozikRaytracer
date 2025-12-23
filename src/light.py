from __future__ import annotations

from functools import cached_property
from typing import Generator
import numpy as np

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
        self.position = Vector3.from_array(position)
        self.color = Vector3.from_array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @cached_property
    def get_position(self) -> Vector3:
        return self.position

    def samples(self, direction: Vector3) -> Generator[Vector3]:
        t, b = _build_orthogonal_basis(direction.normalized, self.radius)
        top_left = self.position - (t + b) / 2

        n = int(Scene().settings.root_number_shadow_rays)

        tl_data = top_left._data
        t_data = t._data / n
        b_data = b._data / n

        x_indices, y_indices = np.meshgrid(np.arange(n), np.arange(n))

        x_indices = x_indices.flatten()
        y_indices = y_indices.flatten()

        jitter_x = np.random.random(n * n)
        jitter_y = np.random.random(n * n)

        total_x = (x_indices + jitter_x)[:, np.newaxis] * t_data
        total_y = (y_indices + jitter_y)[:, np.newaxis] * b_data

        all_points = tl_data + total_x + total_y

        for point in all_points:
            yield Vector3(point[0], point[1], point[2])