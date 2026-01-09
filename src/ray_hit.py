from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from surfaces.surface import Surface
from material import Material


@dataclass
class RayHit:
    surface: Surface
    point: np.array
    normal: np.array
    material: Material or int
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
