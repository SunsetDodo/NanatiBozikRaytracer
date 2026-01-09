from __future__ import annotations
from typing import Optional


class Surface:
    def get_hit(self, ray: 'Ray', scene: 'Scene') -> Optional['RayHit']:
        raise NotImplementedError()

    def hit_distance(self, ray: "Ray", t_min: float, t_max: float) -> Optional[float]:
        """
        Fast intersection query used for shadow/occlusion rays.
        Implementations should avoid allocating RayHit / computing normals.
        Returns the closest t in (t_min, t_max), or None.
        """
        return None

    def bounding_box(self):
        """
        Returns an AABB (from bvh.AABB) if the surface is finite, else None.
        BVH construction will ignore surfaces that return None.
        """
        return None

