import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scene import Scene, SceneSingleton
from scene_settings import SceneSettings
from ray import Ray, trace_ray
from vector3 import Vector3
from material import Material
from light import Light


BACKGROUND = [0.1, 0.2, 0.3]


def reset_scene(max_bounce_depth=5):
    SceneSingleton.instance = None
    s = Scene()
    s.settings = SceneSettings(BACKGROUND, 1, max_bounce_depth)
    s.materials = []
    s.lights = []
    s.surfaces = []
    return s


def test_scene_settings_stores_max_bounce_depth():
    settings = SceneSettings([0, 0, 0], 1, 7)
    assert settings.max_bounce_depth == 7


def test_scene_settings_default_max_bounce_depth():
    settings = SceneSettings([0, 0, 0], 1)
    assert settings.max_bounce_depth == 5


def test_depth_zero_returns_background_color():
    reset_scene(max_bounce_depth=0)
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    result = trace_ray(ray, depth=0)
    expected = Vector3.from_array(BACKGROUND)
    assert abs(result.x - expected.x) < 1e-9
    assert abs(result.y - expected.y) < 1e-9
    assert abs(result.z - expected.z) < 1e-9


def test_depth_negative_returns_background_color():
    reset_scene()
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    result = trace_ray(ray, depth=-1)
    expected = Vector3.from_array(BACKGROUND)
    assert abs(result.x - expected.x) < 1e-9
    assert abs(result.y - expected.y) < 1e-9
    assert abs(result.z - expected.z) < 1e-9


def test_empty_scene_returns_background_at_any_depth():
    for depth in (0, 1, 5, 10):
        reset_scene(max_bounce_depth=depth)
        ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
        result = trace_ray(ray, depth=depth)
        expected = Vector3.from_array(BACKGROUND)
        assert abs(result.x - expected.x) < 1e-9, f"failed at depth={depth}"


def test_no_recursion_error_at_depth_5():
    reset_scene(max_bounce_depth=5)
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    try:
        trace_ray(ray, depth=5)
    except RecursionError:
        pytest.fail("RecursionError raised despite depth guard")


def test_max_bounce_depth_passed_to_trace_ray():
    """scene_settings.max_bounce_depth drives the initial depth argument."""
    s = reset_scene(max_bounce_depth=3)
    assert s.settings.max_bounce_depth == 3
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    result = trace_ray(ray, depth=s.settings.max_bounce_depth)
    assert result is not None
