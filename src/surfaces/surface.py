from __future__ import annotations
from typing import Optional


class Surface:
    def get_hit(self, ray: 'Ray', scene: 'Scene') -> Optional['RayHit']:
        raise NotImplementedError()

