import numpy as np

class StructuredTensor:
    def __init__(self, src, size=10.0, sharpness=1.0, eccentricity=0.5):
        height, width, _ = src.shape
        self.height = height
        self.width = width
        print(height)
        self.dst = np.zeros(shape=(height, width, 3))
        self.size = size
        self.sharpness = sharpness
        self.eccentricity = eccentricity
        self.src = src

    @staticmethod
    def square(x):
        return x * x
    
    def getDst(self):
        return self.dst

    def process(self, pos):
        # The weight kernels of the filter optimized for rotational symmetry 
        # described in section "3.2.1 Gradient Calculation".
        corner_weight = 0.182
        center_weight = 1.0 - 2.0 * corner_weight
        
        # Pad the input image to handle edge cases
        
        lowX = 0 if pos[0]-1 < 0 else pos[0]-1
        lowY = 0 if pos[1]-1 < 0 else pos[1]-1
        heyY = min( pos[1] + 1, self.height - 1)
        heyX = min (pos[0] + 1, self.width - 1)


        
        # Compute partial derivatives
        x_partial_derivative = (
            -corner_weight * self.src[heyY, lowX] +
            -center_weight * self.src[pos[1],lowX] +
            -corner_weight * self.src[ lowY, lowX] +
            corner_weight * self.src[ heyY, heyX] +
            center_weight * self.src[ pos[1], heyX] +
            corner_weight * self.src[ lowY, heyX]
        )
        
        y_partial_derivative = (
            corner_weight * self.src[ heyY, lowX] +
            center_weight * self.src[ heyY, pos[0]] +
            corner_weight * self.src[ heyY, heyX] +
            -corner_weight * self.src[lowY, lowX] +
            -center_weight * self.src[ lowY, pos[0]] +
            -corner_weight * self.src[ lowY, heyX]
        )
        
        # Compute structure tensor components
        dxdx = np.sum(x_partial_derivative * x_partial_derivative, axis=-1)
        dxdy = np.sum(x_partial_derivative * y_partial_derivative, axis=-1)
        dydy = np.sum(y_partial_derivative * y_partial_derivative, axis=-1)
        
        # Stack the components into a 3-channel image
        self.dst[pos[1], pos[0]]=np.array([dxdx, dxdy, dydy])
    

