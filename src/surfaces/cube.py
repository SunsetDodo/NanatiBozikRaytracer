from typing import Optional

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from vector3 import Vector3, element_min, element_max
from .surface import Surface


class Cube(Surface):
    def __init__(self, position, scale, material_index):
        self.position = Vector3.from_array(position)
        self.scale = scale
        self.material_index = material_index

    def get_hit(self, ray: 'Ray') -> Optional['RayHit']:
        inv_dir = ray.direction.inverse()

        half_size = self.scale * 0.5
        offset = Vector3(half_size, half_size, half_size)
        min_pt = self.position - offset
        max_pt = self.position + offset

        t1 = (min_pt - ray.origin) * inv_dir
        t2 = (max_pt - ray.origin) * inv_dir

        t_min_vec = element_min(t1, t2)
        t_max_vec = element_max(t1, t2)

        t_enter = t_min_vec.max_component()
        t_exit = t_max_vec.min_component()

        if t_exit < t_enter or t_exit < EPSILON:
            return None

        t = t_enter if t_enter > EPSILON else t_exit
        hit_point = ray.origin + (ray.direction * t)

        normal = Vector3.zero()
        diff = hit_point - self.position

        if abs(diff.x / half_size) - 1 > EPSILON:
            normal.x = -1 if diff.x < 0 else 1
        elif abs(diff.y / half_size) - 1 > EPSILON:
            normal.y = -1 if diff.y < 0 else 1
        else:
            normal.z = -1 if diff.z < 0 else 1

        return RayHit(self, hit_point, normal, self.material_index, t)
