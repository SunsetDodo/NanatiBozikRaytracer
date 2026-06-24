from light import Light
from ray_hit import RayHit
from vector3 import Vector3, dot, vec3_convolution
from surfaces.surface import Surface
from scene import Scene

import heapq
from typing import List, Optional


class Ray:
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def at(self, distance: float):
        return self.origin + self.direction * distance


def find_hit(ray, max_list_depth: int) -> List[RayHit]:
    hits_heap = []

    for surface in Scene().surfaces:
        # TODO - consider sending max t (after we fill max_list_depth) to get_hit and stop if we reached t.
        ray_hit = surface.get_hit(ray)
        if ray_hit is not None:
            entry = (-ray_hit.distance, id(ray_hit), ray_hit)
            if len(hits_heap) < max_list_depth:
                heapq.heappush(hits_heap, entry)
            else:
                furthest_dist_in_heap = -hits_heap[0][0]
                if ray_hit.distance < furthest_dist_in_heap:
                    heapq.heapreplace(hits_heap, entry)

    result_hits = [item[2] for item in hits_heap]
    result_hits.sort(key=lambda h: h.distance)

    return result_hits


def trace_ray(ray: Ray, depth: int = 5) -> Vector3:
    if depth <= 0:
        return Vector3.from_array(Scene().settings.background_color)

    hit_list = find_hit(ray, 10)
    if not hit_list:
        return Vector3.from_array(Scene().settings.background_color)

    closest_hit = hit_list[0]

    color = Vector3.zero()

    for light in Scene().lights:
        total_samples = Scene().settings.root_number_shadow_rays ** 2
        reachable_samples = 0

        light_vector = light.get_position - closest_hit.point
        light_dir = light_vector.normalized

        for sample in light.samples(
                light_vector):
            origin = closest_hit.point + closest_hit.normal * Scene.EPSILON

            sample_vector = sample - origin
            shadow_ray = Ray(origin, sample_vector)

            if not is_occluded(shadow_ray, sample_vector.length):
                reachable_samples += 1

        visibility = reachable_samples / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - visibility)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=(ray.direction * -1).normalized,
            light_dir=light_dir
        ) * shadow
        color += color_contrib

    if closest_hit.material.reflection_color.length > 0:
        d = ray.direction.normalized
        n = closest_hit.normal
        reflect_dir = d - n * (2 * dot(d, n))
        reflected_ray = Ray(closest_hit.point + n * Scene.EPSILON, reflect_dir)
        color += vec3_convolution(closest_hit.material.reflection_color, trace_ray(reflected_ray, depth - 1))

    if closest_hit.material.transparency > 0:
        transmit_ray = Ray(closest_hit.point - closest_hit.normal * Scene.EPSILON, ray.direction)
        transmit_color = trace_ray(transmit_ray, depth - 1)
        color = color * (1 - closest_hit.material.transparency) + transmit_color * closest_hit.material.transparency

    return color.clamp_01()


def is_occluded(ray, max_distance: float) -> bool:
    for surface in Scene().surfaces:
        hit = surface.get_hit(ray)
        if hit is not None and hit.distance < max_distance - Scene.EPSILON:
            return True
    return False