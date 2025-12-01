from surface import Surface

class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index
