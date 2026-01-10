from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from bvh import BVHNode
from consts import EPSILON

if TYPE_CHECKING:
    from ray import Ray
    from ray_hit import RayHit

class Scene:
    settings: Optional['SceneSettings']
    camera: Optional['Camera']
    surfaces: List['Surface']
    finite_surfaces: List['Surface']
    infinite_surfaces: List['Surface']
    materials: List['Material']
    lights: List['Light']
    bvh: Optional[BVHNode]

    # Hardcoded constants
    EPSILON = 1e-9
    INF = float("inf")

    def __init__(self):
        self.settings = None
        self.camera = None
        self.surfaces = []
        self.finite_surfaces = []
        self.infinite_surfaces = []
        self.materials = []
        self.lights = []
        self.bvh = None
        self.advanced_shadows = False
        self.estimate_reflections = False

    def background_color(self):
        return self.settings.background_color_np

    def build_acceleration(self) -> None:
        self.finite_surfaces = []
        self.infinite_surfaces = []

        for s in self.surfaces:
            if getattr(s, "bounding_box", None) is None:
                self.infinite_surfaces.append(s)
                continue
            bbox = s.bounding_box()
            if bbox is None:
                self.infinite_surfaces.append(s)
            else:
                self.finite_surfaces.append(s)

        self.bvh = BVHNode.build(self.finite_surfaces) if self.finite_surfaces else None

    def closest_hit(self, ray: "Ray", t_min: float = EPSILON, t_max: float = INF) -> Optional["RayHit"]:
        best_hit: Optional[RayHit] = None

        if self.bvh is not None:
            best_hit = self.bvh.hit_closest(ray, self, t_min=t_min, t_max=t_max)
            if best_hit is not None:
                t_max = min(t_max, best_hit.distance)

        # Checking infinite surfaces seperately as they dont fit in AABB
        for s in self.infinite_surfaces:
            hit = s.get_hit(ray, self)
            if hit is None:
                continue
            if hit.distance < t_min or hit.distance > t_max:
                continue
            best_hit = hit
            t_max = hit.distance

        return best_hit

    def any_hit(self, ray: "Ray", t_min: float, t_max: float) -> bool:
        if self.bvh is not None and self.bvh.hit_any(ray, self, t_min=t_min, t_max=t_max):
            return True

        for s in self.infinite_surfaces:
            fn = getattr(s, "hit_distance", None)
            if fn is not None:
                if fn(ray, t_min, t_max) is not None:
                    return True
                continue

            hit = s.get_hit(ray, self)
            if hit is not None and (t_min < hit.distance < t_max):
                return True

        return False
