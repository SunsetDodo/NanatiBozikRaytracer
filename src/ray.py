from src.math import Vector3


class Ray:
    def __init__(self, origin: Vector3, target: Vector3, ttl: int):
        self.origin = origin
        self.target = target