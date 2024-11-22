import numpy as np
from multiprocessing import Pool
from .Filter import Filter
class Kuwahara(Filter):
    # implementation of kuwahara filter without anisotropy
    def __init__(self, structure_tensor: np.ndarray, src: np.ndarray=None, size: float=10.0, sharpness: float=1.0, eccentricity: float=0.5):
        self.size = size
        self.sharpness = sharpness
        self.eccentricity = eccentricity
        self.structure_tensor = structure_tensor
        if src is not None:
            height, width, _ = src.shape
            self.height = height
            self.width = width
            self.dst = np.zeros(shape=(height, width, 4))
            self.src = src

    @staticmethod
    def square(x: float) -> float:
        return x * x

    def process(self, pos: list[int]) -> np.ndarray:
        return  self.src[pos[0], pos[1]]

