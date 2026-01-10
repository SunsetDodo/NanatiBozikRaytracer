from __future__ import annotations

from typing import Optional, Sequence
from ray import Ray
from ray_hit import RayHit

import numpy as np

from consts import EPSILON


def intersect_aabb(ray: Ray, min_pt: np.ndarray, max_pt: np.ndarray):
    t1 = (min_pt - ray.origin) * ray.inv_direction
    t2 = (max_pt - ray.origin) * ray.inv_direction

    t_min_vec = np.minimum(t1, t2)
    t_max_vec = np.maximum(t1, t2)

    t_enter = float(t_min_vec.max())
    t_exit = float(t_max_vec.min())

    return t_enter, t_exit


class AABB:
    __slots__ = ("min_point", "max_point")

    def __init__(self, minimum, maximum):
        self.min_point = np.array(minimum, dtype=np.float64)
        self.max_point = np.array(maximum, dtype=np.float64)

    def centroid(self) -> np.ndarray:
        return (self.min_point + self.max_point) * 0.5

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:

        t_enter, t_exit = intersect_aabb(ray, self.min_point, self.max_point)

        if t_exit < t_enter:
            return False

        if t_exit < t_min or t_enter > t_max:
            return False

        return True


def surrounding_box(a: AABB, b: AABB) -> AABB:
    new_min = np.minimum(a.min_point, b.min_point)
    new_max = np.maximum(a.max_point, b.max_point)
    return AABB(new_min, new_max)


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

