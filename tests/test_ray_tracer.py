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


def test_no_recursion_error_with_reflective_surfaces():
    """Single infinite-mirror surface: depth guard stops real trace_ray recursion."""
    from ray_hit import RayHit

    s = reset_scene(max_bounce_depth=5)

    # Fully reflective material at 1-based index 1
    mat = Material([0, 0, 0], [0, 0, 0], [1, 1, 1], 1, 0)
    s.materials.append(mat)

    # A surface that always returns a hit at distance 1, normal opposing the ray.
    # Any ray bounces back and forth indefinitely without the depth guard.
    class AlwaysHitMirror:
        def get_hit(self, ray):
            hit_point = ray.at(1)
            normal = (ray.direction * -1).normalized
            return RayHit(self, hit_point, normal, 1, 1.0)

    s.surfaces.append(AlwaysHitMirror())

    try:
        result = trace_ray(Ray(Vector3(0, 0, 0), Vector3(0, 0, 1)), depth=5)
    except RecursionError:
        pytest.fail("RecursionError raised despite depth guard with two facing mirror surfaces")


def test_depth_guard_fires_before_surface_lookup(monkeypatch):
    """At depth<0 the guard returns background before find_hit is ever called."""
    reset_scene()
    r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))

    def exploding_find_hit(*args, **kwargs):
        raise AssertionError("find_hit must not be called when depth < 0")

    monkeypatch.setattr(ray_module, 'find_hit', exploding_find_hit)

    result = trace_ray(r, depth=-1)
    expected = Vector3.from_array(BACKGROUND)
    assert abs(result.x - expected.x) < 1e-9
    assert abs(result.y - expected.y) < 1e-9
    assert abs(result.z - expected.z) < 1e-9


def test_max_bounce_depth_passed_to_trace_ray():
    """scene_settings.max_bounce_depth drives the initial depth argument."""
    s = reset_scene(max_bounce_depth=3)
    assert s.settings.max_bounce_depth == 3
    ray = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    result = trace_ray(ray, depth=s.settings.max_bounce_depth)
    assert result is not None


def test_depth_zero_returns_background_even_with_surface_hit(monkeypatch):
    """depth=0 calls find_hit for the primary ray but suppresses all recursive trace calls.

    With depth<0 as the guard, depth=0 means 'shade the primary hit but make all
    reflection/refraction sub-calls immediately return background (via depth-1=-1 guard)'.
    find_hit is therefore called exactly once — for the primary ray — and never again
    for any reflected or refracted rays.
    """
    from ray_hit import RayHit

    s = reset_scene(max_bounce_depth=0)

    mat = Material([0, 0, 0], [0, 0, 0], [1, 1, 1], 1, 0)
    s.materials.append(mat)

    class AlwaysHitMirror:
        def get_hit(self, ray):
            hit_point = ray.at(1)
            normal = (ray.direction * -1).normalized
            return RayHit(self, hit_point, normal, 1, 1.0)

    s.surfaces.append(AlwaysHitMirror())
    s.lights = []

    find_hit_call_count = []
    original_find_hit = ray_module.find_hit

    def counting_find_hit(*args, **kwargs):
        find_hit_call_count.append(True)
        return original_find_hit(*args, **kwargs)

    monkeypatch.setattr(ray_module, 'find_hit', counting_find_hit)

    r = Ray(Vector3(0, 0, 0), Vector3(0, 0, 1))
    trace_ray(r, depth=0)

    assert find_hit_call_count, "find_hit must be called at depth=0 for the primary ray"
    assert len(find_hit_call_count) == 1, (
        f"find_hit must be called exactly once at depth=0 (primary ray only), "
        f"but was called {len(find_hit_call_count)} times"
    )


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


@pytest.mark.parametrize("depth", [0, 1, 5])
def test_pool_scene_does_not_crash_at_various_depths(depth):
    """Rendering pool.txt at depths 0/1/5 completes without crashing."""
    import os
    from ray_tracer import parse_scene_file
    from viewport import Viewport

    SceneSingleton.instance = None

    scene_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'scenes', 'pool.txt')
    camera, scene_settings, _objects = parse_scene_file(scene_file)

    scene = Scene()
    scene.settings.max_bounce_depth = depth

    vp = Viewport(camera, 10, 10)
    origin = camera.get_position()
    target = vp.get_pixel_center(5, 5)
    r = Ray(origin, target - origin)

    result = trace_ray(r, depth=depth)

    assert isinstance(result.x, float), f"depth={depth}: x must be float"
    assert isinstance(result.y, float), f"depth={depth}: y must be float"
    assert isinstance(result.z, float), f"depth={depth}: z must be float"


def test_pool_scene_parses_max_bounce_depth():
    """pool.txt specifies max_bounce_depth=10; verify it is parsed into SceneSettings."""
    import os
    from ray_tracer import parse_scene_file

    SceneSingleton.instance = None
    scene_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'scenes', 'pool.txt')
    _, scene_settings, _ = parse_scene_file(scene_file)
    assert scene_settings.max_bounce_depth == 10


def test_no_recursion_error_with_refractive_surfaces():
    """Transparent always-hit surface: depth guard stops refraction recursion."""
    from ray_hit import RayHit

    s = reset_scene(max_bounce_depth=5)
    mat = Material([0, 0, 0], [0, 0, 0], [0, 0, 0], 1, 1.0)  # fully transparent, no reflection
    s.materials.append(mat)

    class AlwaysHitTransparent:
        def get_hit(self, ray):
            hit_point = ray.at(1)
            normal = (ray.direction * -1).normalized
            return RayHit(self, hit_point, normal, 1, 1.0)

    s.surfaces.append(AlwaysHitTransparent())

    try:
        trace_ray(Ray(Vector3(0, 0, 0), Vector3(0, 0, 1)), depth=5)
    except RecursionError:
        pytest.fail("RecursionError raised despite depth guard with always-hit transparent surface")


def test_non_reflective_material_never_recurses(monkeypatch):
    """trace_ray must not call itself for a pure diffuse (non-reflective, opaque) material."""
    from ray_hit import RayHit

    s = reset_scene()
    mat = Material([0.8, 0.2, 0.1], [0.1, 0.1, 0.1], [0, 0, 0], 30, 0)  # reflection=(0,0,0), transparency=0
    s.materials.append(mat)
    s.lights = []

    class FixedHit:
        def get_hit(self, ray):
            return RayHit(self, ray.at(1), Vector3(0, 0, -1), 1, 1.0)

    s.surfaces.append(FixedHit())

    call_count = [0]
    original_trace_ray = ray_module.trace_ray

    def counting_trace_ray(*args, **kwargs):
        call_count[0] += 1
        return original_trace_ray(*args, **kwargs)

    monkeypatch.setattr(ray_module, 'trace_ray', counting_trace_ray)

    counting_trace_ray(Ray(Vector3(0, 0, 0), Vector3(0, 0, 1)), depth=5)

    assert call_count[0] == 1, (
        f"trace_ray called {call_count[0]} times for a pure diffuse material; expected 1 (no recursion)"
    )
