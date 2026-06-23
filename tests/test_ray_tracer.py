import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import ray as ray_module
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


def test_no_recursion_error_at_depth_5(monkeypatch):
    """Simulation: depth guard prevents infinite recursion from mutually-facing mirrors.

    NOTE: This is a simulation test.  Because reflection call-sites in trace_ray
    are currently `pass` stubs, trace_ray never actually recurses.  This test
    replaces trace_ray with mirror_trace (which does recurse through the patched
    module-level name) to exercise that a depth guard *can* stop runaway recursion.
    The guard validated here belongs to mirror_trace, not to the production
    trace_ray.  Restructure once the reflection branch is implemented so the real
    trace_ray guard is exercised end-to-end.
    """
    reset_scene(max_bounce_depth=5)
    r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))

    def mirror_trace(ray, depth=5):
        if depth <= 0:
            return Vector3.from_array(Scene().settings.background_color)
        # Recurse through the module-level name so the monkeypatch keeps the
        # loop alive — simulating two mirrors facing each other indefinitely.
        return ray_module.trace_ray(ray, depth - 1)

    monkeypatch.setattr(ray_module, 'trace_ray', mirror_trace)

    try:
        result = mirror_trace(r, depth=5)
    except RecursionError:
        pytest.fail("RecursionError raised despite depth guard")


def test_depth_guard_fires_before_surface_lookup(monkeypatch):
    """At depth<=0 the guard returns background before find_hit is ever called."""
    for exhausted_depth in (0, -1):
        reset_scene()
        r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))

        def exploding_find_hit(*args, **kwargs):
            raise AssertionError(
                f"find_hit must not be called when depth <= 0 (got depth={exhausted_depth})"
            )

        monkeypatch.setattr(ray_module, 'find_hit', exploding_find_hit)

        result = trace_ray(r, depth=exhausted_depth)
        expected = Vector3.from_array(BACKGROUND)
        assert abs(result.x - expected.x) < 1e-9, f"depth={exhausted_depth}"
        assert abs(result.y - expected.y) < 1e-9, f"depth={exhausted_depth}"
        assert abs(result.z - expected.z) < 1e-9, f"depth={exhausted_depth}"


def test_max_bounce_depth_passed_to_trace_ray():
    """scene_settings.max_bounce_depth drives the initial depth argument."""
    s = reset_scene(max_bounce_depth=3)
    assert s.settings.max_bounce_depth == 3
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    result = trace_ray(ray, depth=s.settings.max_bounce_depth)
    assert result is not None


def test_depth_zero_returns_background_even_with_surface_hit(monkeypatch):
    """depth=0 must short-circuit to background *before* shading any hit surface.

    The guard must fire before find_hit, not after — otherwise a depth=0 call
    would still shade the surface (wrong) even though zero bounces remain.
    This test simulates a scene where find_hit would return a valid hit; if the
    guard is placed incorrectly (or uses depth < 0 instead of depth <= 0) the
    test catches it because find_hit raising would surface the bug.
    """
    reset_scene(max_bounce_depth=0)
    r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))

    def would_hit(*args, **kwargs):
        raise AssertionError("find_hit must not be reached at depth=0")

    monkeypatch.setattr(ray_module, 'find_hit', would_hit)

    result = trace_ray(r, depth=0)
    expected = Vector3.from_array(BACKGROUND)
    assert abs(result.x - expected.x) < 1e-9
    assert abs(result.y - expected.y) < 1e-9
    assert abs(result.z - expected.z) < 1e-9


def test_positive_depth_does_call_find_hit(monkeypatch):
    """Sanity check: at depth=1 the guard does NOT fire and find_hit is called."""
    reset_scene(max_bounce_depth=1)
    r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    find_hit_called = []

    original_find_hit = ray_module.find_hit

    def tracking_find_hit(*args, **kwargs):
        find_hit_called.append(True)
        return original_find_hit(*args, **kwargs)

    monkeypatch.setattr(ray_module, 'find_hit', tracking_find_hit)
    trace_ray(r, depth=1)
    assert find_hit_called, "find_hit must be called when depth > 0"
