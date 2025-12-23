import logging
import random

from vector3 import Vector3, cross
from camera import Camera

from math import atan, pi, tan

class Viewport:
    def __init__(self, camera: Camera, image_width: int, image_height: int):
        aspect_ratio = image_width / image_height

        self.logger = logging.getLogger("Raytracer").getChild("Viewport")
        self.width = camera.screen_width
        self.horizontal_fov = 2 * atan(self.width / (2 * camera.screen_distance))
        self.logger.debug(
            "Viewport width: %.2f. Horizontal FOV: %.3f Rads (%.2f degrees).",
            self.horizontal_fov,
            self.horizontal_fov * 180 / pi,
            self.width
        )

        self.height = camera.screen_width / aspect_ratio
        self.vertical_fov = 2 * atan(self.height / (2 * camera.screen_distance))
        self.logger.debug(
            "Viewport height: %.2f. Vertical FOV: %.3f Rads (%.2f degrees).",
            self.vertical_fov,
            self.vertical_fov * 180 / pi,
            self.height
        )

        # Calculating right and down vectors of the viewport
        forward = (camera.get_look_at() - camera.get_position()).normalized
        right = -cross(forward, Vector3.from_array(camera.up_vector)).normalized
        down = -cross(forward, right).normalized  # we cross again in case configured up and look at are not perpendicular

        self.u = right * self.width
        self.v = down * self.height

        self.delta_u = self.u / image_width
        self.delta_v = self.v / image_height

        center = camera.get_position() + forward * camera.screen_distance
        self.top_left = center - self.u / 2 - self.v / 2
        self.start_pixel = self.top_left + (self.delta_u + self.delta_v) / 2

    def get_pixel_center(self, x, y) -> Vector3:
        return self.start_pixel + self.delta_u * x + self.delta_v * y

    def get_random_location_in_pixel(self, x, y):
        return self.top_left + self.delta_u * (x + random.random()) + self.delta_v * (y + random.random())
