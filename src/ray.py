import numpy as np

from consts import EPSILON

class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction
        self.inv_direction = 1.0 / np.where(direction == 0, EPSILON, direction)

        self.ox = float(origin[0])
        self.oy = float(origin[1])
        self.oz = float(origin[2])
        self.dx = float(direction[0])
        self.dy = float(direction[1])
        self.dz = float(direction[2])

    def at(self, distance: float):
        return self.origin + self.direction * distance


def trace_ray(scene, ray, max_recursion_depth: int = 10, processed_objects: set = None):
    if processed_objects is None:
        processed_objects = set()

    if max_recursion_depth == -1:
        return scene.settings.background_color

    closest_hit = scene.closest_hit(ray, t_min=EPSILON)
    if closest_hit is None:
        return scene.settings.background_color

    background = scene.settings.background_color
    diffuse_spec = np.zeros(3, dtype=np.float64)
    reflection = np.zeros(3, dtype=np.float64)

    if closest_hit.material.transparency > 0:
        if scene.process_inner:
            new_origin = closest_hit.point + ray.direction * EPSILON
        else:
            new_origin = closest_hit.point + ray.direction * closest_hit.skip_distance

        new_ray = Ray(new_origin, ray.direction)
        background = trace_ray(scene, new_ray, max_recursion_depth - 1)

    if np.any(closest_hit.material.reflection_color):
        reflect_dir = ray.direction - 2 * (ray.direction @ closest_hit.normal) * closest_hit.normal
        reflect_ray = Ray(closest_hit.point + closest_hit.normal * EPSILON, reflect_dir)
        reflection = trace_ray(scene, reflect_ray, max_recursion_depth - 1) * closest_hit.material.reflection_color

    for light in scene.lights:
        total_samples = scene.settings.root_number_shadow_rays ** 2

        light_vector = light.position - closest_hit.point
        light_dir = light_vector / np.linalg.norm(light_vector)

        light_hits = 0
        for sample in light.samples(light_dir, scene):
            origin = closest_hit.point + closest_hit.normal * EPSILON

            shadow_ray = Ray(origin, sample - origin)
            if scene.advanced_shadows:
                accumulated_transparency = 1.0
                current_t = EPSILON

                for _ in range(max_recursion_depth):
                    hit = scene.closest_hit(shadow_ray, t_min=current_t)
                    if hit is None:
                        break

                    current_t = hit.distance + EPSILON
                    if not scene.process_inner and hit.surface.obj_id in processed_objects:
                        continue

                    accumulated_transparency *= hit.material.transparency
                    processed_objects.add(hit.surface.obj_id)

                    if accumulated_transparency == 0.0:
                        break
                else:
                    accumulated_transparency = 0.0

                light_hits += accumulated_transparency
            else:
                light_hits += 1.0 - is_occluded(scene, shadow_ray, float(np.linalg.norm(sample - origin)))

        average_light_contrib = light_hits / total_samples
        shadow = 1.0 - light.shadow_intensity * (1.0 - average_light_contrib)

        color_contrib = closest_hit.material.calculate_light(
            light=light,
            normal_dir=closest_hit.normal,
            view_dir=(ray.direction * -1) / np.linalg.norm(ray.direction),
            light_dir=light_dir,
            estimate=scene.estimate_reflections
        ) * shadow
        diffuse_spec += color_contrib

    color = closest_hit.material.transparency * background + (
            1 - closest_hit.material.transparency) * diffuse_spec + reflection

    return np.clip(color, 0.0, 1.0)


def is_occluded(scene, ray, max_distance: float) -> bool:
    return scene.any_hit(ray, t_min=EPSILON, t_max=max_distance - EPSILON)