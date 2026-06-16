"""Regression tests: existing scenes must still parse without error."""
import os
import pytest
from scene import Scene
from conftest import _reset_scene

SCENES_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'scenes')


def _parse(filename):
    # The autouse reset_scene fixture already set up a clean Scene; we just
    # need parse_scene_file to populate it from the file.
    from ray_tracer import parse_scene_file
    return parse_scene_file(os.path.join(SCENES_DIR, filename))


def test_pool_scene_parses():
    # reset_scene fixture adds 1 material; then pool.txt adds 7 → 8 total
    # Re-reset here so pool.txt starts from a completely clean slate.
    _reset_scene()
    camera, settings, objects = _parse('pool.txt')
    s = Scene()
    assert len(s.surfaces) == 7   # 6 spheres + 1 plane
    assert len(s.materials) == 7
    assert len(s.lights) == 5


def test_teapot_scene_parses():
    _reset_scene()
    camera, settings, objects = _parse('teapot.txt')
    s = Scene()
    # pyramid.obj has 6 faces → 6 Triangle surfaces
    assert len(s.surfaces) == 6
    assert len(s.materials) == 1
    assert len(s.lights) == 1
