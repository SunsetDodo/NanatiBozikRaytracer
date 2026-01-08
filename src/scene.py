from __future__ import annotations
from typing import List, Optional

class Scene:
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
        return self.settings.background_color_np
