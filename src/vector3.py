from __future__ import annotations

from typing import List

class Vector3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, item):
        if isinstance(item, int) and 0 <= item <= 2:
            return [self.x, self.y, self.z][item]
        raise ValueError("Can only get numbers between 0 and 2")

    @staticmethod
    def from_array(array: List[float]) -> Vector3:
        if len(array) != 3:
            raise ValueError("Vector3 must have 3 elements")
        return Vector3(array[0], array[1], array[2])

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0, 0, 0)
