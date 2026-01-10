from __future__ import annotations
from typing import Optional


class Surface:
    def get_hit(self, ray: 'Ray', scene: 'Scene') -> Optional['RayHit']:
        raise NotImplementedError()

    def hit_distance(self, ray: "Ray", t_min: float, t_max: float) -> Optional[float]:
        return None

    def bounding_box(self):
        return None

