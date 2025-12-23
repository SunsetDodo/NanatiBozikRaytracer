from __future__ import annotations
from functools import cached_property
from typing import List, Optional

from vector3 import Vector3


class SceneSingleton(type):
    instance = None
    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance


class Scene(metaclass=SceneSingleton):
    settings: Optional['SceneSettings']
    camera: Optional['Camera']
    surfaces: List['Surface']
    materials: List['Material']
    lights: List['Light']

    # Hardcoded constants
    EPSILON = 1e-9

    def __init__(self):
        self.settings = None
        self.camera = None
        self.surfaces = []
        self.materials = []
        self.lights = []

    def background_color(self):
        return Vector3.from_array(self.settings.background_color)
