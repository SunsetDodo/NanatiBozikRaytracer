import numpy as np

def normalize(v: np.array) -> np.array:
    return v / np.linalg.norm(v)