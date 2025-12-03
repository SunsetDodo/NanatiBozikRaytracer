from functools import cached_property

from src.vector3 import Vector3


class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @cached_property
    def get_position(self):
        return Vector3.from_array(self.position)
