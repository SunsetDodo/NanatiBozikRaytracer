from typing import Optional

from src.consts import EPSILON
from src.ray import Ray
from src.ray_hit import RayHit
from src.vector3 import Vector3, dot
from src.scene import Scene
from .surface import Surface



class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = normal
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

        return RayHit(self, hit_point, Vector3.from_array(self.normal), self.material_index, t)
