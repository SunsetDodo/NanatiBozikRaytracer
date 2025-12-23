from typing import Optional

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from vector3 import Vector3, dot
from scene import Scene
from .surface import Surface


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = Vector3.from_array(normal)
        self.offset = offset
        self.material_index = material_index

    def get_hit(self, ray: 'Ray') -> Optional['RayHit']:
        dprod = dot(ray.direction, self.normal)
        if abs(dprod) < EPSILON:
            return None

        t = (self.offset - dot(ray.origin, self.normal)) / dprod
        if t < EPSILON:
            return None
        hit_point = ray.origin + (ray.direction * t)

        return RayHit(self, hit_point, self.normal, self.material_index, t)
