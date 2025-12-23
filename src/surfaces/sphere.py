from math import sqrt
from typing import Optional

from src.consts import EPSILON
from src.ray import Ray
from src.ray_hit import RayHit
from src.vector3 import Vector3
from .surface import Surface
import numpy as np  # Make sure numpy is imported


class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position = Vector3.from_array(position)
        self.radius = radius
        self.material_index = material_index

    def get_hit(self, ray: 'Ray') -> Optional['RayHit']:

        O = ray.origin._data
        D = ray.direction._data
        P = self.position._data

        L = P - O

        a = D @ D

        b = -2.0 * (D @ L)

        c = (L @ L) - (self.radius ** 2)

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
        normal = (hit_point - self.position).normalized

        return RayHit(self, hit_point, normal, self.material_index, t)