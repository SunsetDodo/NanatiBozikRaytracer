from __future__ import annotations

from functools import cached_property
from typing import Generator

from scene import Scene

import numpy as np

def _build_orthogonal_basis(direction: np.ndarray, radius: float):
    dx, dy, dz = direction

    if abs(dx) > abs(dz):
        tangent = np.array([-dy, dx, 0.0], dtype=direction.dtype)
    else:
        tangent = np.array([0.0, -dz, dy], dtype=direction.dtype)

    # normalize tangent
    t_len = np.linalg.norm(tangent)
    if t_len != 0.0:
        tangent /= t_len

    # bitangent = normalize(cross(direction, tangent))
    bitangent = np.cross(direction, tangent)
    b_len = np.linalg.norm(bitangent)
    if b_len != 0.0:
        bitangent /= b_len

    return tangent * radius, bitangent * radius



class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    import numpy as np

    def samples(self, direction: np.ndarray, scene) -> np.ndarray:
        dir_norm = direction / np.linalg.norm(direction)

        t, b = _build_orthogonal_basis(dir_norm, self.radius)
        top_left = self.position - (t + b) * 0.5

        n = int(scene.settings.root_number_shadow_rays)

        t_step = t / n
        b_step = b / n

        idx = np.arange(n, dtype=np.float64)
        x_idx, y_idx = np.meshgrid(idx, idx, indexing="xy")

        x_jitters = np.random.random((n, n))
        y_jitters = np.random.random((n, n))

        x_offsets = (x_idx + x_jitters)[..., None] * t_step  # (n, n, 3)
        y_offsets = (y_idx + y_jitters)[..., None] * b_step  # (n, n, 3)

        all_points = (top_left + x_offsets + y_offsets).reshape(-1, 3)
        return all_points
