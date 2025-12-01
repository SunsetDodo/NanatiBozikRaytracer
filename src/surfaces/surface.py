from __future__ import annotations
from typing import Optional

from src.ray import Ray
from src.ray_hit import RayHit


class Surface:
    def get_hit(self, ray: Ray) -> Optional[RayHit]:
        raise NotImplementedError()

