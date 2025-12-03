from packaging.version import VersionComparisonMethod

from src.light import Light
from src.ray_hit import RayHit
from src.vector3 import Vector3
from src.surfaces.surface import Surface
from scene import Scene

from typing import List, Optional


class Ray:
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def at(self, distance: float):
        return self.origin + self.direction * distance


def find_hit(ray, closest: bool = False) -> Optional[RayHit]:
    closest_hit = None
    scene = Scene()
    for surface in scene.surfaces:
        ray_hit = surface.get_hit(ray)
        if ray_hit < closest_hit:
            if not closest:
                return ray_hit
            closest_hit = ray_hit

    return closest_hit


def trace_ray(ray, max_recursion_depth: int = 10) -> Vector3:
    closest_hit = find_hit(ray, True)
    if not closest_hit:
        return Scene().background_color

    color = Vector3.zero()

    # Handle Lights and Shadows
    for light in Scene().lights:
        if find_hit(light.get_position() - closest_hit.point, False) is not None:
            continue
        ...

    # Handle Reflection

    # Handle Refraction

    return color



