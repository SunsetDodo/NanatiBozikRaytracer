from __future__ import annotations

import sys

from dataclasses import dataclass

from surfaces.surface import Surface
from src.math import Vector3
from src.material import Material

from typing import Optional

@dataclass
class RayHit:
    surface: Surface
    point: Vector3
    normal: Vector3
    material: Material
    distance: float

    def __gt__(self, other):
        if isinstance(other, RayHit):
            return self.distance > other.distance
        if other is None:
            return True
        raise TypeError("Can only compare RayHit objects")

    def __lt__(self, other):
        if isinstance(other, RayHit):
            return self.distance < other.distance
        if other is None:
            return False
        raise TypeError("Can only compare RayHit objects")

    def __eq__(self, other):
        if isinstance(other, RayHit):
            return self.distance == other.distance
        if other is None:
            return False
        raise TypeError("Can only compare RayHit objects")
