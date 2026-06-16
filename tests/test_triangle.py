import pytest
from vector3 import Vector3
from ray import Ray
from surfaces.triangle import Triangle, load_obj

MATERIAL_IDX = 1  # index 1 → materials[0] in Scene, populated by conftest


def make_triangle():
    """Triangle in the XY plane: v0=(0,0,0), v1=(2,0,0), v2=(0,2,0). Normal=(0,0,1)."""
    return Triangle(Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0), MATERIAL_IDX)


def test_hit_through_centroid():
    tri = make_triangle()
    ray = Ray(Vector3(2 / 3, 2 / 3, -5), Vector3(0, 0, 1))
    hit = tri.get_hit(ray)
    assert hit is not None
    assert abs(hit.distance - 5.0) < 1e-6


def test_miss_outside_triangle():
    tri = make_triangle()
    # u=2.5 → outside triangle bounds
    ray = Ray(Vector3(5, 5, -5), Vector3(0, 0, 1))
    hit = tri.get_hit(ray)
    assert hit is None


def test_miss_ray_aimed_away():
    tri = make_triangle()
    # Ray from behind origin going further away; t will be negative
    ray = Ray(Vector3(1, 1, 5), Vector3(0, 0, 1))
    hit = tri.get_hit(ray)
    assert hit is None


def test_parallel_ray():
    tri = make_triangle()
    # Ray direction lies in the triangle's plane → determinant ≈ 0
    ray = Ray(Vector3(0, 0, 1), Vector3(1, 0, 0))
    hit = tri.get_hit(ray)
    assert hit is None


def test_near_zero_t_rejected():
    tri = make_triangle()
    # Ray origin just above the triangle plane; t < EPSILON so hit is rejected
    ray = Ray(Vector3(2 / 3, 2 / 3, 1e-12), Vector3(0, 0, -1))
    hit = tri.get_hit(ray)
    assert hit is None


def test_edge_boundary_hit():
    tri = make_triangle()
    # Ray aimed at midpoint of edge v0–v1 (1, 0, 0); v=0, u=0.5 — boundary is valid
    ray = Ray(Vector3(1, 0, -5), Vector3(0, 0, 1))
    hit = tri.get_hit(ray)
    assert hit is not None


def test_backface_ray_no_culling():
    tri = make_triangle()
    # Ray from +z side hitting the back face; no culling, should still return a hit
    ray = Ray(Vector3(2 / 3, 2 / 3, 5), Vector3(0, 0, -1))
    hit = tri.get_hit(ray)
    assert hit is not None
    assert abs(hit.distance - 5.0) < 1e-6


def test_load_obj_triangle(tmp_path):
    obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    obj_file = tmp_path / "tri.obj"
    obj_file.write_text(obj_content)
    triangles = load_obj(str(obj_file), MATERIAL_IDX)
    assert len(triangles) == 1
    assert isinstance(triangles[0], Triangle)


def test_load_obj_quad_fan_triangulation(tmp_path):
    obj_content = (
        "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\n"
        "f 1 2 3 4\n"
    )
    obj_file = tmp_path / "quad.obj"
    obj_file.write_text(obj_content)
    triangles = load_obj(str(obj_file), MATERIAL_IDX)
    assert len(triangles) == 2


def test_load_obj_with_texture_indices(tmp_path):
    # OBJ face lines may use "v/vt/vn" notation
    obj_content = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1/1/1 2/2/1 3/3/1\n"
    )
    obj_file = tmp_path / "textured.obj"
    obj_file.write_text(obj_content)
    triangles = load_obj(str(obj_file), MATERIAL_IDX)
    assert len(triangles) == 1
