from typing import Optional

import numpy as np

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from .surface import Surface
from bvh import AABB, intersect_aabb


class Cube(Surface):
    def __init__(self, position, scale, material_index, obj_id):
        self.position = np.array(position)
        self.scale = scale
        self.material_index = material_index
        self.obj_id = obj_id

        half_size = 0.5 * self.scale
        offset = np.array([half_size, half_size, half_size], dtype=float)
        self.min_pt = self.position - offset
        self.max_pt = self.position + offset

    def get_hit(self, ray: "Ray", scene: 'Scene') -> Optional["RayHit"]:
        t_enter, t_exit = intersect_aabb(ray, self.min_pt, self.max_pt)

        if t_exit < t_enter or t_exit < EPSILON:
            return None

        t = t_enter if t_enter > EPSILON else t_exit

        hit_point = ray.origin + ray.direction * t

        diff = hit_point - self.position
        half_size = 0.5 * self.scale

        with np.errstate(divide='ignore'):
            scaled = diff / half_size

        axis = int(np.abs(scaled).argmax())
        normal = np.zeros(3, dtype=self.position.dtype)
        normal[axis] = -1.0 if diff[axis] < 0.0 else 1.0

        thickness = abs(t_enter - t_exit) + EPSILON

        return RayHit(self, hit_point, normal, scene.materials[self.material_index - 1], t, thickness)

    def hit_distance(self, ray: "Ray", t_min: float, t_max: float) -> Optional[float]:
        t_enter, t_exit = intersect_aabb(ray, self.min_pt, self.max_pt)

        if t_exit < t_enter:
            return None

        if t_exit < t_min or t_enter > t_max:
            return None

        if t_min < t_enter < t_max:
            return t_enter
        if t_min < t_exit < t_max:
            return t_exit

        return None

    def bounding_box(self) -> AABB:
        half = 0.5 * float(self.scale)
        offset = np.array([half, half, half], dtype=self.position.dtype)
        return AABB(self.position - offset, self.position + offset)