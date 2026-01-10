from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from bvh import BVHNode
from consts import EPSILON, INF

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

    def __init__(self):
        self.settings = None
        self.camera = None
        self.finite_surfaces = []
        self.infinite_surfaces = []
        self.materials = []
        self.lights = []
        self.bvh = None
        self.advanced_shadows = False
        self.estimate_reflections = False
        self.process_inner = False

    def background_color(self):
        return self.settings.background_color_np

    def build_acceleration(self) -> None:
        self.bvh = BVHNode.build(self.finite_surfaces) if self.finite_surfaces else None

    def closest_hit(self, ray: "Ray", t_min: float = EPSILON, t_max: float = INF) -> Optional["RayHit"]:
        best_hit: Optional[RayHit] = None

        if self.bvh is not None:
            best_hit = self.bvh.hit_closest(ray, self, t_min=t_min, t_max=t_max)
            if best_hit is not None:
                t_max = min(t_max, best_hit.distance)

        # Checking infinite planes seperately as they dont fit in AABB
        for s in self.infinite_surfaces:
            hit = s.get_hit(ray, self)
            if hit is None:
                continue
            if not  t_min < hit.distance < t_max:
                continue
            best_hit = hit
            t_max = hit.distance

        return best_hit

    def any_hit(self, ray: "Ray", t_min: float, t_max: float) -> bool:
        if self.bvh is not None and self.bvh.hit_any(ray, self, t_min=t_min, t_max=t_max):
            return True

        for s in self.infinite_surfaces:
            if s.hit_distance(ray, t_min, t_max) is not None:
                return True

        return False