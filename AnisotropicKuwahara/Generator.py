import numpy as np
import math
from multiprocessing import Pool
from .Filter import Filter

class RainbowGenerator(Filter):
    def __init__(self):
        self.height = 100
        self.width = 200
        self.dst = np.zeros(shape=(self.height, self.width, 4))

    def getDst(self):
        return self.dst

    def process(self, pos):
        return np.array([
            pos[0]*(1/self.height),
            pos[1]*(1/self.width),
            (self.height - pos[0] - 1)*(1/self.height),
            (self.width - pos[1] - 1)*(1/self.width),
        ])
     

class BlotchGenerator(Filter):
    def __init__(self):
        self.height = 100
        self.width = 200
        self.dst = np.zeros(shape=(self.height, self.width, 4))

    def getDst(self):
        return self.dst

    def process(self, pos):
        y = math.floor(pos[0]/8)
        x = math.floor(pos[1]/8)
        print(y,x)
        return np.array([
            y*(8/self.height),
            x*(8/self.width),
            1-(y*(4/self.height)+x*(4/self.width)),
            1,
        ])
     
     