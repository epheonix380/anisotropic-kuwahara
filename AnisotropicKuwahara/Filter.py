import numpy as np
from multiprocessing import Pool

class Filter:
    
    def getDst(self):
        return self.dst
    
    def doX(self, y):
        column = np.zeros(shape=(self.width, 4))
        try:
            for x in range(self.width):
                column[x] = self.process([y, x])
        except Exception:
            print(f"Failure in column {str(y)}")
        return column
    
    def singleProcessAll(self):
        for y in range(self.height):
            self.dst[y] = self.doX(y=y)

    def multiProcessAll(self):
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


    def process(self, pos):
        return np.array([0,0,0,1])