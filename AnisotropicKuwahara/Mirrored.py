import numpy as np
from multiprocessing import Pool
from .Filter import Filter

class Mirrored(Filter):
    def __init__(self, src):
        height, width, _ = src.shape
        self.height = height
        self.width = width
        self.dst = np.zeros(shape=(height, width, 4))
        self.src = src

    def process(self, pos):
        return self.src[pos[0], self.width-pos[1]-1]
     