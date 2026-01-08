from typing import Optional

import numpy as np

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from .surface import Surface


class Cube(Surface):
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index

    def get_hit(self, ray: "Ray", scene: 'Scene') -> Optional["RayHit"]:
        origin = ray.origin
        direction = ray.direction

        # inverse direction (handle zeros via inf)
        with np.errstate(divide="ignore"):
            inv_dir = 1.0 / direction

        half_size = 0.5 * self.scale
        offset = np.array([half_size, half_size, half_size], dtype=origin.dtype)

        min_pt = self.position - offset
        max_pt = self.position + offset

        # slab intersections
        t1 = (min_pt - origin) * inv_dir
        t2 = (max_pt - origin) * inv_dir

        t_min_vec = np.minimum(t1, t2)
        t_max_vec = np.maximum(t1, t2)

        t_enter = t_min_vec.max()
        t_exit = t_max_vec.min()

        if t_exit < t_enter or t_exit < EPSILON:
            return None

        t = t_enter if t_enter > EPSILON else t_exit
        hit_point = origin + direction * t

        # compute normal from dominant axis of |diff|
        diff = hit_point - self.position
        # scale to box size so faces are near ±1
        scaled = diff / half_size
        abs_scaled = np.abs(scaled)

        axis = int(abs_scaled.argmax())
        normal = np.zeros(3, dtype=origin.dtype)
        normal[axis] = -1.0 if diff[axis] < 0.0 else 1.0

        return RayHit(self, hit_point, normal, scene[self.material_index - 1], t)
