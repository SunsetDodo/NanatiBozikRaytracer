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
    for surface in Scene().surfaces:
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
        reacable_samples = 0
        for _ in range(Scene().settings.root_number_shadow_rays):
            to_light = light.get_position - closest_hit.point
            sample = light.sample_position(to_light, Scene().settings.SHADOW_RAY_RADIUS)
            if find_hit(Ray(closest_hit.point, sample - closest_hit.point)) is not None:
                reacable_samples += 1

        if reacable_samples > 0:
            color += (light.calculate_color(closest_hit.material) *
                      (reacable_samples / Scene().settings.root_number_shadow_rays))

    if not max_recursion_depth == 0:
        ...
        # Handle Reflection

        # Handle Refraction

    return color



