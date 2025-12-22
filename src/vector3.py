from __future__ import annotations

from functools import cached_property
from typing import List, Tuple

from math import acos

from consts import EPSILON

class Vector3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self._array = (x, y, z)
        self._x = x
        self._y = y
        self._z = z

    def __getitem__(self, item):
        if isinstance(item, int) and 0 <= item <= 2:
            return self._array[item]
        raise ValueError("Can only get numbers between 0 and 2")

    def __add__(self, other):
        return Vector3.from_array([self[i] + other[i] for i in (0, 1, 2)])

    def __sub__(self, other):
        return Vector3.from_array([self[i] - other[i] for i in (0, 1, 2)])

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only multiply Vector3 with numeric scalar")
        return Vector3.from_array([val * other for val in self._array])

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only divide Vector3 by numeric scalar")
        if other == 0:
            raise ZeroDivisionError("Can't deivide Vector3 by zero")
        return self * (1 / other)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Vec3({self.x},{self.y},{self.z})"

    @cached_property
    def normalized(self):
        return self / self.length

    @cached_property
    def length_squared(self):
        """Returns the distance from (0,0,0) squared"""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    @cached_property
    def length(self):
        return self.length_squared ** 0.5

    @cached_property
    def inverse(self):
        x = 1.0 / self.x if abs(self.x) > EPSILON else float('inf')
        y = 1.0 / self.y if abs(self.y) > EPSILON else float('inf')
        z = 1.0 / self.z if abs(self.z) > EPSILON else float('inf')
        return Vector3(x, y, z)

    def to_tuple(self) -> Tuple[float, float, float]:
        return self._array

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._clear_cached_properties()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self._clear_cached_properties()

    @property
    def z(self):
        return self._z

    def clamp_01(self):
        return Vector3(
            max(min(self.x, 1.0), 0.0),
            max(min(self.y, 1.0), 0.0),
            max(min(self.z, 1.0), 0.0)
        )


    @z.setter
    def z(self, value):
        self._z = value
        self._clear_cached_properties()

    def _clear_cached_properties(self):
        for key in ("normalized", "length_squared", "length"):
            self.__dict__.pop(key, None)

    @staticmethod
    def from_array(array: List[float]) -> Vector3:
        if len(array) != 3:
            raise ValueError("Vector3 must have 3 elements")
        return Vector3(array[0], array[1], array[2])

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0, 0, 0)


    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def min_component(self) -> float:
        return min(self.x, self.y, self.z)


def dot(u: Vector3, v: Vector3) -> float:
    return sum([u[i] * v[i] for i in (0, 1, 2)])


def angle(u: Vector3, v: Vector3) -> float:
    if u.length_squared == 0 or v.length_squared == 0:
        raise ValueError("Cannot compute angle with zero-length vector")

    d = acos(dot(u.normalized, v.normalized))
    d = max(-1.0, min(1.0, d))
    return acos(d)


def cross(u: Vector3, v: Vector3) -> Vector3:
    return Vector3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    )


def element_min(v1: Vector3, v2: Vector3) -> Vector3:
    return Vector3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z))


def element_max(v1: Vector3, v2: Vector3) -> Vector3:
    return Vector3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z))

def vec3_convolution(u: Vector3, v: Vector3) -> Vector3:
    return Vector3(
        u[0] * v[0],
        u[1] * v[1],
        u[2] * v[2]
    )
