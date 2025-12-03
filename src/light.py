import random
from functools import cached_property
from math import pi, cos, sin

from src.material import Material
from src.vector3 import Vector3, cross


def _build_orthogonal_basis(direction: Vector3):

    if abs(direction.x) > abs(direction.z):
        tangent = Vector3(-direction.y, direction.x, 0)
    else:
        tangent = Vector3(0, -direction.z, direction.y)

    tangent = tangent.normalized
    bitangent = cross(direction, tangent).normalized

    return tangent, bitangent


def _random_point_in_disk(radius: float):
    theta = random.uniform(0, 2 * pi)
    r = radius * random.random()
    x = r * cos(theta)
    y = r * sin(theta)

    return x, y

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @cached_property
    def get_position(self) -> Vector3:
        return Vector3.from_array(self.position)

    def sample_position(self, direction: Vector3, radius: float) -> Vector3:
        t, b = _build_orthogonal_basis(direction.normalized)
        offset_x, offset_y = _random_point_in_disk(radius)

        return self.get_position + t * offset_x + b * offset_y

    def calculate_color(self, material: Material) -> Vector3:
        return Vector3.zero()

