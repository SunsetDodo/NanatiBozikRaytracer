from typing import List, Optional

from src.camera import Camera
from src.material import Material
from src.scene_settings import SceneSettings
from src.surfaces.surface import Surface
from src.light import Light


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Scene(metaclass=Singleton):
    settings: Optional[SceneSettings]
    camera: Optional[Camera]
    surfaces: List[Surface]
    materials: List[Material]
    lights: List[Light]

    def __init__(self):
        self.settings = None
        self.camera = None
        self.surfaces = []
        self.materials = []
        self.lights = []