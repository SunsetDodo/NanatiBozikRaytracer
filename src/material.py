from __future__ import annotations

from src.light import Light
from src.vector3 import Vector3


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.shininess = shininess
        self.transparency = transparency

    def calculate_light(self, light: Light) -> Vector3:
        return Vector3.zero()
