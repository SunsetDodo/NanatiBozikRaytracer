import numpy as np

from ray_hit import RayHit
from utils import normalize
from scene import Scene

import heapq
from typing import List, Optional


class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction

    def at(self, distance: float):
        return self.origin + self.direction * distance


def find_hit(ray, max_list_depth: int) -> list[RayHit]:
    surfaces = Scene().surfaces      # ideally pass this in instead of calling Scene()
    hits: list[RayHit] = []

    for surface in surfaces:
        ray_hit = surface.get_hit(ray)
        if ray_hit is None:
            continue

        d = ray_hit.distance

        if len(hits) < max_list_depth:
            hits.append(ray_hit)
        else:
            worst_idx = 0
            worst_dist = hits[0].distance
            for i in range(1, len(hits)):
                dist_i = hits[i].distance
                if dist_i > worst_dist:
                    worst_dist = dist_i
                    worst_idx = i

            if d < worst_dist:
                hits[worst_idx] = ray_hit

    hits.sort(key=lambda h: h.distance)
    return hits



def trace_ray(ray, max_recursion_depth: int = 10) -> np.array:
    if max_recursion_depth == -1:
        return Scene().settings.background_color_np

    hit_list = find_hit(ray, max_recursion_depth)
    if not hit_list:
        return Scene().settings.background_color_np

    closest_hit = hit_list[0]

    color = np.zeros(3, dtype=float)

    for light in Scene().lights:
        total_samples = Scene().settings.root_number_shadow_rays ** 2
        reachable_samples = 0

        light_vector = light.position - closest_hit.point
        light_dir = normalize(light_vector)

        for sample in light.samples(
                light_vector):
            origin = closest_hit.point + closest_hit.normal * Scene.EPSILON

            shadow_ray = Ray(origin, sample - origin)

            if not is_occluded(shadow_ray, max_recursion_depth):
                reachable_samples += 1

        visibility = reachable_samples / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - visibility)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=normalize(ray.direction * -1),
            light_dir=light_dir
        ) * shadow
        color += color_contrib

    if closest_hit.material.transparency > 0:
        pass

    return np.clip(color, 0.0, 1.0)

def is_occluded(ray, max_distance: float) -> bool:
    for surface in Scene().surfaces:
        hit = surface.get_hit(ray)
        if hit is not None and hit.distance < max_distance - Scene.EPSILON:
            return True
    return False