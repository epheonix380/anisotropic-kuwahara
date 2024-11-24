import numpy as np
from .Filter import Filter
import cv2
from enum import Enum
class GradientMethod(Enum):
    JAHNE = "jahne"
    SOBEL = "sobel"

class StructuredTensor:
    def __init__(self, src, size=10.0, sharpness=1.0, eccentricity=0.5):
        """_summary_

        Args:
            src (_type_): _description_
            size (float, optional): _description_. Defaults to 10.0.
            sharpness (float, optional): _description_. Defaults to 1.0.
            eccentricity (float, optional): _description_. Defaults to 0.5.
        """
        height, width, _ = src.shape
        self.height = height
        self.width = width
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
    
    def getGradientsRGB(self, cv_dtype=-1, method:GradientMethod=GradientMethod.JAHNE)->np.ndarray:
        """Given image, return dxdx, dydy and dxdy gradients. Image must be in RGBA format
        Args:
            cv_dtype (int, optional): cv2 data type. Defaults to -1.
            method (GradientMethod, optional): the method to use for gradient calculation. Defaults to GradientMethod.JAHNE.

        Returns:
            _type_: array with shape (height, width, 3) where the 3 channels are dxdx, dydy, dxdy
        """

        # jahne's gradient kernel as described in section "3.2.1 Gradient Calculation" of kyprianidis paper (2011)
        p1 = 0.182
        kernelx = np.array([
            [p1,0,-p1],
            [1-2*p1,0,2*p1-1],
            [p1,0,-p1]])
        kernelx = kernelx / 2

        if method == GradientMethod.SOBEL:
            kernelx = np.array([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])

        kernely = kernelx.T


        r, g, b, a = cv2.split(self.src)
        rx = cv2.filter2D(r,cv_dtype, kernelx)
        ry = cv2.filter2D(r, cv_dtype, kernely)
        rxrx = rx * rx
        ryry = ry * ry
        rxry = rx * ry

        gx = cv2.filter2D(g, cv_dtype, kernelx)
        gy = cv2.filter2D(g, cv_dtype, kernely)
        gxgx = gx * gx
        gygy = gy * gy
        gxgy = gx * gy

        bx = cv2.filter2D(b, cv_dtype, kernelx)
        by = cv2.filter2D(b, cv_dtype, kernely)
        bxby = bx * by
        byby = by * by
        bxbx = bx * bx

        ax = cv2.filter2D(a, cv_dtype, kernelx)
        ay = cv2.filter2D(a, cv_dtype, kernely)
        axax = ax * ax
        ayay = ay * ay
        axay = ax * ay

        dxdx = rxrx + gxgx + bxbx + axax
        dydy = ryry + gygy + byby + ayay
        dxdy = rxry + gxgy + bxby + axay

        return cv2.merge([dxdx, dydy, dxdy])
    
    def getGradientGray(self, method:GradientMethod=GradientMethod.JAHNE):
        """Given image, calculate gradient based on grayscale vers of image. return dxdx, dydy and dxdy gradients. Image must be in RGBA format
        Args:
            method (GradientMethod, optional): the method to use for gradient calculation. Defaults to GradientMethod.JAHNE.
        Returns:
            _type_: array with shape (height, width, 3) where the 3 channels are dxdx, dydy, dxdy
        """
        # jahne's gradient kernel as described in section "3.2.1 Gradient Calculation" of kyprianidis paper (2011)
        p1 = 0.182
        kernelx = np.array([
            [p1,0,-p1],
            [1-2*p1,0,2*p1-1],
            [p1,0,-p1]])
        kernelx = kernelx / 2

        if method == GradientMethod.SOBEL:
            kernelx = np.array([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])

        kernely = kernelx.T

        # cvtColor 	only accepts these 8-bit unsigned, 16-bit unsigned ( CV_16UC... )  float(32) so we need to convert to float32
        if self.src.dtype == np.uint8 or self.src.dtype == np.uint16 or self.src.dtype == np.float32:
            grayscale = cv2.cvtColor(self.src, cv2.COLOR_RGBA2GRAY).astype(np.float32)
        elif self.src.dtype == np.float64:
            converted = np.array(self.src, dtype=np.float32)
            grayscale = cv2.cvtColor(converted, cv2.COLOR_RGBA2GRAY).astype(np.float64)
        else:
            raise ValueError("Unsupported dtype")
        

        dx = cv2.filter2D(grayscale, -1, kernelx)
        dy = cv2.filter2D(grayscale, -1, kernely)
        dxdx = dx * dx
        dydy = dy * dy
        dxdy = dx * dy

        return cv2.merge([dxdx, dydy, dxdy])
    

    def getGradientGraySeparated(self):
        """Faster implementation of getGradientGrayscale. Given image, calculate gradient based on grayscale vers of image. Image must be in RGBA format
        Args:
            method (GradientMethod, optional): the method to use for gradient calculation. Defaults to GradientMethod.JAHNE.
        Returns:
            _type_: array with shape (height, width, 3) where the 3 channels are dxdx, dydy, dxdy gradients
        """
        # jahne's gradient kernel as described in section "3.2.1 Gradient Calculation" of kyprianidis paper (2011)
        p1 = 0.182
        kernel_x_tall = np.array([[p1],[1-2*p1],[p1]]) / 2  # happy to rename this if someone has a better idea
        kernel_x_long = np.array([[1,0,-1]])

        kernel_y_tall = kernel_x_long.T
        kernel_y_long = kernel_x_tall.T

        # cvtColor 	only accepts these 8-bit unsigned, 16-bit unsigned ( CV_16UC... )  float(32) so we need to convert to float32
        if self.src.dtype == np.uint8 or self.src.dtype == np.uint16 or self.src.dtype == np.float32:
            grayscale = cv2.cvtColor(self.src, cv2.COLOR_RGBA2GRAY).astype(np.float32)
        elif self.src.dtype == np.float64:
            converted = np.array(self.src, dtype=np.float32)
            grayscale = cv2.cvtColor(converted, cv2.COLOR_RGBA2GRAY).astype(np.float64)
        else:
            raise ValueError("Unsupported dtype")
        
        # dx = cv2.filter2D(cv2.filter2D(grayscale, -1, kernel_x_long), -1, kernel_x_tall)
        dx = cv2.filter2D(grayscale, -1, kernel_x_long)
        dx = cv2.filter2D(dx, -1, kernel_x_tall)

        # dy = cv2.filter2D(cv2.filter2D(grayscale, -1, kernel_y_long), -1, kernel_y_tall)
        dy = cv2.filter2D(grayscale, -1, kernel_y_long)
        dy = cv2.filter2D(dy, -1, kernel_y_tall)
        dxdx = dx * dx
        dydy = dy * dy
        dxdy = dx * dy

        return cv2.merge([dxdx, dydy, dxdy])

   
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
    

