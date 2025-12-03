from src.vector3 import Vector3


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width

    def get_position(self) -> Vector3:
        return Vector3.from_array(self.position)

    def get_look_at(self) -> Vector3:
        return Vector3.from_array(self.look_at)
