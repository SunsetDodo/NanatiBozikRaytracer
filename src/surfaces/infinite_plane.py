from typing import Optional

import numpy as np

from consts import EPSILON
from ray import Ray
from ray_hit import RayHit
from .surface import Surface


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.offset = offset
        self.material_index = material_index

    def get_hit(self, ray: 'Ray', scene: 'Scene') -> Optional['RayHit']:
        d_prod = ray.direction @ self.normal
        if abs(d_prod) < EPSILON:
            return None

        t = (self.offset - (ray.origin @ self.normal)) / d_prod
        if t < EPSILON:
            return None
        hit_point = ray.origin + (ray.direction * t)

        return RayHit(self, hit_point, self.normal, scene.materials[self.material_index - 1], t)
