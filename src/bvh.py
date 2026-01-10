from __future__ import annotations

from typing import Optional, Sequence
from ray import Ray
from ray_hit import RayHit

import numpy as np

from consts import EPSILON


class AABB:
    __slots__ = ("min0", "min1", "min2", "max0", "max1", "max2")

    def __init__(self, minimum, maximum):
        self.min0 = float(minimum[0])
        self.min1 = float(minimum[1])
        self.min2 = float(minimum[2])
        self.max0 = float(maximum[0])
        self.max1 = float(maximum[1])
        self.max2 = float(maximum[2])

    def centroid(self) -> np.ndarray:
        return np.array(
            [
                (self.min0 + self.max0) * 0.5,
                (self.min1 + self.max1) * 0.5,
                (self.min2 + self.max2) * 0.5,
            ],
            dtype=np.float64,
        )

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        ox, oy, oz = ray.ox, ray.oy, ray.oz
        dx, dy, dz = ray.dx, ray.dy, ray.dz

        # X
        inv_d = float("inf") if dx == 0.0 else (1.0 / dx)
        t0 = (self.min0 - ox) * inv_d
        t1 = (self.max0 - ox) * inv_d
        if inv_d < 0.0:
            t0, t1 = t1, t0
        if t0 > t_min:
            t_min = t0
        if t1 < t_max:
            t_max = t1
        if t_max <= t_min:
            return False

        # Y
        inv_d = float("inf") if dy == 0.0 else (1.0 / dy)
        t0 = (self.min1 - oy) * inv_d
        t1 = (self.max1 - oy) * inv_d
        if inv_d < 0.0:
            t0, t1 = t1, t0
        if t0 > t_min:
            t_min = t0
        if t1 < t_max:
            t_max = t1
        if t_max <= t_min:
            return False

        # Z
        inv_d = float("inf") if dz == 0.0 else (1.0 / dz)
        t0 = (self.min2 - oz) * inv_d
        t1 = (self.max2 - oz) * inv_d
        if inv_d < 0.0:
            t0, t1 = t1, t0
        if t0 > t_min:
            t_min = t0
        if t1 < t_max:
            t_max = t1
        if t_max <= t_min:
            return False

        return True


def surrounding_box(a: AABB, b: AABB) -> AABB:
    return AABB(
        (min(a.min0, b.min0), min(a.min1, b.min1), min(a.min2, b.min2)),
        (max(a.max0, b.max0), max(a.max1, b.max1), max(a.max2, b.max2)),
    )


def build_range(items_range) -> BVHNode:
    n = len(items_range)
    if n == 1:
        s, bbox, _ = items_range[0]
        return BVHNode(aabb=bbox, surface=s)
    if n == 2:
        left = build_range(items_range[:1])
        right = build_range(items_range[1:])
        return BVHNode(aabb=surrounding_box(left.aabb, right.aabb), left=left, right=right)

    centroids = np.stack([c for (_, _, c) in items_range], axis=0)
    axis = int(np.argmax(np.var(centroids, axis=0)))

    items_sorted = sorted(items_range, key=lambda it: float(it[2][axis]))
    mid = n // 2
    left = build_range(items_sorted[:mid])
    right = build_range(items_sorted[mid:])
    return BVHNode(aabb=surrounding_box(left.aabb, right.aabb), left=left, right=right)


class BVHNode:
    __slots__ = ("aabb", "left", "right", "surface")

    def __init__(
        self,
        aabb: AABB,
        left: Optional["BVHNode"] = None,
        right: Optional["BVHNode"] = None,
        surface=None,
    ):
        self.aabb = aabb
        self.left = left
        self.right = right
        self.surface = surface

    @staticmethod
    def build(surfaces: Sequence) -> Optional["BVHNode"]:
        if not surfaces:
            return None

        items = []
        for s in surfaces:
            bbox = s.bounding_box()
            if bbox is None:
                continue
            centroid = bbox.centroid()
            items.append((s, bbox, centroid))

        if not items:
            return None

        return build_range(items)

    def hit_closest(self, ray: Ray, scene, t_min: float = EPSILON, t_max: float = float("inf")) -> Optional[RayHit]:
        if not self.aabb.hit(ray, t_min, t_max):
            return None

        if self.surface is not None:
            hit = self.surface.get_hit(ray, scene)
            if hit is None:
                return None
            if hit.distance < t_min or hit.distance > t_max:
                return None
            return hit

        left_hit = self.left.hit_closest(ray, scene, t_min, t_max) if self.left is not None else None
        if left_hit is not None:
            t_max = min(t_max, left_hit.distance)

        right_hit = self.right.hit_closest(ray, scene, t_min, t_max) if self.right is not None else None

        if left_hit and right_hit:
            return left_hit if left_hit.distance < right_hit.distance else right_hit
        return left_hit or right_hit

    def hit_any(self, ray: Ray, scene, t_min: float, t_max: float) -> bool:
        if not self.aabb.hit(ray, t_min, t_max):
            return False

        if self.surface is not None:
            t = self.surface.hit_distance(ray, t_min, t_max)
            return t is not None

        if self.left is not None and self.left.hit_any(ray, scene, t_min, t_max):
            return True
        if self.right is not None and self.right.hit_any(ray, scene, t_min, t_max):
            return True
        return False

