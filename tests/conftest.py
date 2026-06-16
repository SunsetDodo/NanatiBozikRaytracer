import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from scene import Scene, SceneSingleton
from material import Material


def _reset_scene():
    # SceneSingleton stores the instance on the Scene class itself, not on the
    # metaclass. We must set Scene.instance (not SceneSingleton.instance) to
    # force creation of a fresh Scene on the next Scene() call.
    Scene.instance = None


@pytest.fixture(autouse=True)
def reset_scene():
    _reset_scene()
    s = Scene()
    s.materials.append(Material([0.8, 0.8, 0.8], [1, 1, 1], [0, 0, 0], 10, 0))
    yield
    _reset_scene()
