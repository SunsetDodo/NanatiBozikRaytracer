class Vector3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, item):
        if isinstance(item, int) and 0 <= item <= 2:
            return [self.x, self.y, self.z][item]
        raise ValueError("Can only get numbers between 0 and 2")
