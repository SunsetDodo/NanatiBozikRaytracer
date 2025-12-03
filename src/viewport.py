import logging

from src.vector3 import Vector3
from camera import Camera

from math import atan, pi

class Viewport:
    def __init__(self, camera: Camera):
        self.logger = logging.getLogger("Raytracer").getChild("Viewport")
        self.fov = 2 * atan(camera.screen_width / (2 * camera.screen_distance))
        self.logger.debug("Calculated Viewport FOV: %.3f Rads (%.2f degrees)", self.fov, self.fov * 180 / pi)


    def get_pixel_center(self, x, y) -> Vector3:
        return Vector3(0, 0, 0)
