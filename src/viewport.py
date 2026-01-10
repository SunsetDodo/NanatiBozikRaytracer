import random

import numpy as np
from camera import Camera
from utils import normalize

from math import atan

class Viewport:
    def __init__(self, camera: Camera, image_width: int, image_height: int):
        aspect_ratio = image_width / image_height

        self.width = camera.screen_width
        self.horizontal_fov = 2 * atan(self.width / (2 * camera.screen_distance))

        self.height = camera.screen_width / aspect_ratio
        self.vertical_fov = 2 * atan(self.height / (2 * camera.screen_distance))

        # Calculating right and down vectors of the viewport
        forward = normalize(camera.look_at - camera.position)
        right = -normalize(np.cross(forward, camera.up_vector))
        down = -normalize(np.cross(forward, right))

        self.u = right * self.width
        self.v = down * self.height

        self.delta_u = self.u / image_width
        self.delta_v = self.v / image_height

        center = camera.position + forward * camera.screen_distance
        self.top_left = center - self.u / 2 - self.v / 2
        self.start_pixel = self.top_left

    def get_pixel_center(self, x, y) -> np.ndarray:
        return self.start_pixel + self.delta_u * x + self.delta_v * y

    def get_random_location_in_pixel(self, x, y) -> np.ndarray:
        return self.top_left + self.delta_u * (x + random.random()) + self.delta_v * (y + random.random())
