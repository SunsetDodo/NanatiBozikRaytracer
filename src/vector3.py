from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union


class Vector3:
    __slots__ = ['_data']

    def __init__(self, x: float, y: float, z: float):
        # We use a numpy array for internal storage
        self._data = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self):
        return self._data[0]

    @x.setter
    def x(self, v):
        self._data[0] = v

    @property
    def y(self):
        return self._data[1]

    @y.setter
    def y(self, v):
        self._data[1] = v

    @property
    def z(self):
        return self._data[2]

    @z.setter
    def z(self, v):
        self._data[2] = v

    def __getitem__(self, item):
        return self._data[item]

    def __add__(self, other: Vector3):
        res = Vector3(0, 0, 0)
        res._data = self._data + other._data
        return res

    def __sub__(self, other: Vector3):
        res = Vector3(0, 0, 0)
        res._data = self._data - other._data
        return res

    def __mul__(self, other: Union[int, float]):
        if not isinstance(other, (int, float)):
            raise ValueError("Can only multiply Vector3 with numeric scalar")
        res = Vector3(0, 0, 0)
        res._data = self._data * other
        return res

    def __truediv__(self, other: Union[int, float]):
        if other == 0:
            raise ZeroDivisionError("Can't divide Vector3 by zero")
        res = Vector3(0, 0, 0)
        res._data = self._data / other
        return res

    def __neg__(self):
        res = Vector3(0, 0, 0)
        res._data = -self._data
        return res

    def __repr__(self):
        return f"Vec3({self.x:.2f},{self.y:.2f},{self.z:.2f})"

    @property
    def normalized(self):
        norm = np.linalg.norm(self._data)
        if norm == 0:
            return Vector3(0, 0, 0)
        return self / norm

    @property
    def length_squared(self):
        return np.dot(self._data, self._data)

    @property
    def length(self):
        return np.linalg.norm(self._data)

    @property
    def inverse(self):
        # Avoid division by zero issues by using infinity
        with np.errstate(divide='ignore'):
            inv = 1.0 / self._data
        # Replace infinites if needed, or keep numpy behavior
        return Vector3.from_array(inv)

    def to_tuple(self) -> Tuple[float, float, float]:
        return tuple(self._data)

    def clamp_01(self):
        clamped = np.clip(self._data, 0.0, 1.0)
        return Vector3.from_array(clamped)

    def max_component(self) -> float:
        return np.max(self._data)

    def min_component(self) -> float:
        return np.min(self._data)

    @staticmethod
    def from_array(array: Union[List[float], np.ndarray]) -> Vector3:
        if len(array) != 3:
            raise ValueError("Vector3 must have 3 elements")
        return Vector3(array[0], array[1], array[2])

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0, 0, 0)


# Helper functions
def dot(u: Vector3, v: Vector3) -> float:
    return float(u._data @ v._data)


def cross(u: Vector3, v: Vector3) -> Vector3:
    res_data = np.cross(u._data, v._data)
    return Vector3.from_array(res_data)


def vec3_convolution(u: Vector3, v: Vector3) -> Vector3:
    res = Vector3(0, 0, 0)
    res._data = u._data * v._data
    return res


def element_min(v1: Vector3, v2: Vector3) -> Vector3:
    res = Vector3(0, 0, 0)
    res._data = np.minimum(v1._data, v2._data)
    return res


def element_max(v1: Vector3, v2: Vector3) -> Vector3:
    res = Vector3(0, 0, 0)
    res._data = np.maximum(v1._data, v2._data)
    return res
