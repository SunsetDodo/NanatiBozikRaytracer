from typing import Optional

from consts import EPSILON
from ray_hit import RayHit
from vector3 import Vector3, cross, dot
from .surface import Surface


class Triangle(Surface):
    def __init__(self, v0: Vector3, v1: Vector3, v2: Vector3, material_index: int):
        self.v0, self.v1, self.v2 = v0, v1, v2
        self.material_index = material_index
        self._edge1 = v1 - v0
        self._edge2 = v2 - v0
        self.normal = cross(self._edge1, self._edge2).normalized

    def get_hit(self, ray) -> Optional['RayHit']:
        h = cross(ray.direction, self._edge2)
        det = dot(self._edge1, h)
        if abs(det) < EPSILON:
            return None
        inv_det = 1.0 / det
        s = ray.origin - self.v0
        u = inv_det * dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = cross(s, self._edge1)
        v = inv_det * dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = inv_det * dot(self._edge2, q)
        if t < EPSILON:
            return None
        hit_point = ray.at(t)
        return RayHit(self, hit_point, self.normal, self.material_index, t)


def load_obj(path: str, material_index: int) -> list:
    """Parse a Wavefront OBJ file and return a flat list of Triangle instances.

    Supports v and f lines. Non-triangular faces are fan-triangulated.
    Known limitation: negative (relative) OBJ indices are not supported.
    """
    vertices = []
    triangles = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'v':
                vertices.append(Vector3(float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'f':
                # OBJ indices are 1-based; strip texture/normal suffixes (e.g. "1/2/3" → 1)
                indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                for i in range(1, len(indices) - 1):
                    triangles.append(Triangle(
                        vertices[indices[0]],
                        vertices[indices[i]],
                        vertices[indices[i + 1]],
                        material_index
                    ))
    return triangles
