from src.light import Light
from src.math import Vector3
from src.surfaces.surface import Surface

from typing import List


class Ray:
    def __init__(self, origin: Vector3, target: Vector3, ttl: int):
        self.origin = origin
        self.target = target
        self.ttl = ttl


def ray_cast(ray, surfaces: List[Surface], lights: List[Light]):
    closest_hit = None
    for surface in surfaces:
        ray_hit = surface.get_hit(ray)
        if ray_hit < closest_hit:
            closest_hit = ray_hit


