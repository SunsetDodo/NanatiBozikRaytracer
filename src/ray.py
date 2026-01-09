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
        # Cache float components for extremely hot AABB tests (BVH traversal).
        self.ox = float(origin[0])
        self.oy = float(origin[1])
        self.oz = float(origin[2])
        self.dx = float(direction[0])
        self.dy = float(direction[1])
        self.dz = float(direction[2])

    def at(self, distance: float):
        return self.origin + self.direction * distance


def find_hit(scene, ray, max_list_depth: int) -> list[RayHit]:
    hits: list[RayHit] = []

    if max_list_depth == 0:
        return hits

    # Collect consecutive hits along the ray, useful for transparency stacks.
    # (Most shading uses only the closest hit, but this preserves the previous API.)
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


def trace_ray(scene, ray, max_recursion_depth: int = 10) -> np.array:
    if max_recursion_depth == -1:
        return scene.settings.background_color

    hit_list = find_hit(scene, ray, max_recursion_depth)

    if len(hit_list) == 0:
        return scene.settings.background_color

    closest_hit = hit_list[0]

    # TODO - transparency is a scalar while reflection works separately for each channel, in theory if any of the channels are 0 we can skip their calculations.
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
        reachable_samples = 0

        light_vector = light.position - closest_hit.point
        light_dir = normalize(light_vector)

        for sample in light.samples(light_vector, scene):
            origin = closest_hit.point + closest_hit.normal * Scene.EPSILON

            shadow_ray = Ray(origin, sample - origin)

            if not is_occluded(scene, shadow_ray, 1.0):
                reachable_samples += 1

        visibility = reachable_samples / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - visibility)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=normalize(ray.direction * -1),
            light_dir=light_dir
        ) * shadow
        diffuse_spec += color_contrib

    color = closest_hit.material.transparency * background + (
            1 - closest_hit.material.transparency) * diffuse_spec + reflection

    return np.clip(color, 0.0, 1.0)


def is_occluded(scene, ray, max_distance: float) -> bool:
    return scene.any_hit(ray, t_min=EPSILON, t_max=max_distance - Scene.EPSILON)