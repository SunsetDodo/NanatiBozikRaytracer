from src.light import Light
from src.vector3 import Vector3
from src.surfaces.surface import Surface
from scene import Scene

from typing import List


class Ray:
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def at(self, distance: float):
        return self.origin + self.direction * distance


def trace_ray(ray, max_recursion_depth: int = 10):
    closest_hit = None
    scene = Scene()
    for surface in scene.surfaces:
        ray_hit = surface.get_hit(ray)
        if ray_hit < closest_hit:
            closest_hit = ray_hit


