import logging

from src.vector3 import Vector3
from camera import Camera

from math import atan, pi, tan

class Viewport:
    def __init__(self, camera: Camera, aspect_ratio: float):
        self.logger = logging.getLogger("Raytracer").getChild("Viewport")
        self.horizontal_fov = 2 * atan(camera.screen_width / (2 * camera.screen_distance))
        self.logger.debug("Calculated Horizontal FOV: %.3f Rads (%.2f degrees)", self.horizontal_fov, self.horizontal_fov * 180 / pi)

        screen_height = camera.screen_width / aspect_ratio
        self.vertical_fov = 2 * atan(screen_height / (2 * camera.screen_distance))
        self.logger.debug("Calculated Vertical FOV: %.3f Rads (%.2f degrees)", self.vertical_fov, self.vertical_fov * 180 / pi)


    def get_pixel_center(self, x, y) -> Vector3:
        return Vector3(0, 0, 0)
