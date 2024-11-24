import numpy as np
from typing import Union
import cupy as cp
from multiprocessing import Pool

class Filter:

    def setSrc(self, src: Union[np.ndarray, cp.ndarray]):
        self.src = cp.asarray(src)
        height, width, _ = self.src.shape
        self.height = height
        self.width = width
        self.dst = cp.zeros(shape=(height, width, 4))

    
    def getDst(self):
        return self.dst
    
    def doX(self, y: int) -> np.ndarray:
        """
        Helper function to process the image
        """
        column = np.zeros(shape=(self.width, 4))
        try:
            for x in range(self.width):
                column[x] = self.process([y, x])
        except Exception:
            print(f"Failure in column {str(y)}")
        return column
    
    def singleProcessAll(self):
        """
        Starts to process the image. 
        This runs the process function on all pixes in src and saves all the results in dst
        """
        for y in range(self.height):
            self.dst[y] = self.doX(y=y)

    def multiProcessAll(self):
        """
        Starts a multiprocessing pool to process the image. 
        This runs the process function on all pixes in src and saves all the results in dst
        """
        pool = Pool()
        results = []
        for y in range(self.height):
            print(f"Adding {str(y)} to pool")
            results.append(pool.apply_async(self.doX, args=(y,)))
        y = 0
        for result in results:
            res = result.get()
            self.dst[y] = res
            print(f"Got result for {str(y)}")
            y += 1


    def process(self, pos: list[int]) -> cp.ndarray:
        """given the position of a pixel, return the new pixel value
        Returns:
            _type_: RGBa Value of the pixel
        """
        return self.src[pos[0], pos[1]]