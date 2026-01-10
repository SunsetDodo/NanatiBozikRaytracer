import numpy as np

from ray_hit import RayHit
from consts import EPSILON
from utils import normalize
from scene import Scene

from typing import List, Optional


class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction

        # Fast access to ray components
        self.ox = float(origin[0])
        self.oy = float(origin[1])
        self.oz = float(origin[2])
        self.dx = float(direction[0])
        self.dy = float(direction[1])
        self.dz = float(direction[2])

    def at(self, distance: float):
        return self.origin + self.direction * distance


def hit_list(scene, ray, max_list_depth: int) -> list[RayHit]:
    hits: list[RayHit] = []

    if max_list_depth == 0:
        return hits

    current_ray = ray
    traveled = 0.0
    for _ in range(max_list_depth):
        hit = scene.closest_hit(current_ray, t_min=EPSILON)
        if hit is None:
            break

        # Convert hit.distance to be relative to the original ray origin.
        hit.distance += traveled
        hits.append(hit)

        traveled = hit.distance
        current_ray = Ray(hit.point + current_ray.direction * Scene.EPSILON, current_ray.direction)

    return hits


def trace_ray(scene, ray, max_recursion_depth: int = 10):
    if max_recursion_depth == -1:
        return scene.settings.background_color

    closest_hit = scene.closest_hit(ray, t_min=EPSILON)
    if closest_hit is None:
        return scene.settings.background_color

    background = scene.settings.background_color
    diffuse_spec = np.zeros(3, dtype=float)
    reflection = np.zeros(3, dtype=float)

    if closest_hit.material.transparency > 0:
        new_origin = closest_hit.point + ray.direction * Scene.EPSILON
        new_ray = Ray(new_origin, ray.direction)
        background = trace_ray(scene, new_ray, max_recursion_depth - 1)

    if np.any(closest_hit.material.reflection_color):
        reflect_dir = ray.direction - 2 * (ray.direction @ closest_hit.normal) * closest_hit.normal
        reflect_ray = Ray(closest_hit.point + closest_hit.normal * EPSILON, reflect_dir)
        reflection = trace_ray(scene, reflect_ray, max_recursion_depth - 1) * closest_hit.material.reflection_color

    for light in scene.lights:
        total_samples = scene.settings.root_number_shadow_rays ** 2

        light_vector = light.position - closest_hit.point
        light_dir = normalize(light_vector)

        light_hits = 0
        for sample in light.samples(light_vector, scene):
            origin = closest_hit.point + closest_hit.normal * Scene.EPSILON

            shadow_ray = Ray(origin, sample - origin)
            if scene.advanced_shadows:
                accumulated_transparency = 1.0

                hits = hit_list(scene, shadow_ray, max_recursion_depth - 1)
                for hit in hits:
                    accumulated_transparency *= hit.material.transparency

                light_hits += accumulated_transparency
            else:
                light_hits += 1.0 - is_occluded(scene, shadow_ray, float(np.linalg.norm(sample - origin)))

        average_light_contrib = light_hits / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - average_light_contrib)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=normalize(ray.direction * -1),
            light_dir=light_dir,
            estimate=scene.estimate_reflections
        ) * shadow
        diffuse_spec += color_contrib

    color = closest_hit.material.transparency * background + (
            1 - closest_hit.material.transparency) * diffuse_spec + reflection

    return np.clip(color, 0.0, 1.0)


def is_occluded(scene, ray, max_distance: float) -> bool:
    return scene.any_hit(ray, t_min=EPSILON, t_max=max_distance - Scene.EPSILON)