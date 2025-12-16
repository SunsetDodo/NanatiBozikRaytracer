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


def find_hit(ray, max_list_depth: int) -> List[RayHit]:
    hits = []
    for surface in Scene().surfaces:
        ray_hit = surface.get_hit(ray)
        if ray_hit is not None:
            hits.append(ray_hit)

    # TODO - this can be implemented more efficiently by only appending the max_list_depth closest hits
    return sorted(hits)[:max_list_depth]


def trace_ray(ray, max_recursion_depth: int = 10) -> Vector3:
    if max_recursion_depth == -1:
        return Scene().background_color

    hit_list = find_hit(ray, max_recursion_depth)
    if not hit_list:
        return Scene().background_color

    # TODO - implement bonus
    closest_hit = hit_list[0]

    color = Vector3.zero()

    # Handle Lights and Shadows
    for light in Scene().lights:
        total_samples = Scene.settings.root_number_shadow_rays ** 2
        reachable_samples = 0

        for sample in light.samples(light.get_position - closest_hit.point):
            origin = closest_hit.point + closest_hit.normal * Scene.EPSILON
            shadow_ray = Ray(origin, sample - origin)
            if find_hit(shadow_ray) is not None:
                reachable_samples += 1

        visibility = reachable_samples / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - visibility)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=ray.direction.normalize(),
            light_dir=light.get_position - closest_hit.point
        ) * shadow
        color += color_contrib

    if closest_hit.material.transparency > 0:
        pass


    return color.clamp_01()



