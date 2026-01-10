from math import sqrt
from typing import Optional

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from .surface import Surface
import numpy as np

from utils import normalize
from bvh import AABB


class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position =np.array(position)
        self.radius = float(radius)
        self.material_index = material_index

    def get_hit(self, ray: 'Ray', scene) -> Optional['RayHit']:
        look_at = self.position - ray.origin

        a = ray.direction @ ray.direction
        b = -2.0 * (ray.direction @ look_at)
        c = (look_at @ look_at) - (self.radius ** 2)

        discriminant = (b ** 2) - (4 * a * c)

        if discriminant < 0:
            return None

        sqrt_disc = sqrt(discriminant)
        t0 = (-b - sqrt_disc) / (2 * a)
        t1 = (-b + sqrt_disc) / (2 * a)

        if t0 > EPSILON:
            t = t0
        elif t1 > EPSILON:
            t = t1
        else:
            return None

        hit_point = ray.at(t)
        normal = hit_point - self.position
        normal /= np.linalg.norm(normal)

        return RayHit(self, hit_point, normal, scene.materials[self.material_index - 1], t, abs(t0 - t1) + EPSILON)

    def hit_distance(self, ray: "Ray", t_min: float, t_max: float) -> Optional[float]:
        look_at = self.position - ray.origin

        a = ray.direction @ ray.direction
        b = -2.0 * (ray.direction @ look_at)
        c = (look_at @ look_at) - (self.radius ** 2)

        discriminant = (b ** 2) - (4 * a * c)
        if discriminant < 0:
            return None

        sqrt_disc = sqrt(discriminant)
        inv_2a = 1.0 / (2.0 * a)
        t0 = (-b - sqrt_disc) * inv_2a
        if t_min < t0 < t_max:
            return t0

        t1 = (-b + sqrt_disc) * inv_2a
        if t_min < t1 < t_max:
            return t1

        return None

    def bounding_box(self) -> AABB:
        offset = np.array([self.radius, self.radius, self.radius], dtype=self.position.dtype)
        return AABB(self.position - offset, self.position + offset)