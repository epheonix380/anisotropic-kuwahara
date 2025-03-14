import numpy as np
from multiprocessing import Pool
import math
import cv2
import asyncio
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.utils.ImageUtil import *
from AnisotropicKuwahara.utils.TimeReport import TimeReport


class Kuwahara():
    """
    Gaussian Box Kuwahara filter implementation
    """
    num_quadrants = 4

    def __init__(self, kernel_radius:int=5):
        """initialize the Kuwahara filter with a given kernel radius. The signma value of the gaussian is 
        set to kernel_radius/6
        Args:
            kernel_radius (int, optional): radius of kernel. Defaults to 5.
        """

        self.kernel_radius = kernel_radius
        self.size = kernel_radius * 2 + 1

        gauss_1d = cv2.getGaussianKernel(self.size, kernel_radius/3)
        gauss_2d = np.outer(gauss_1d, gauss_1d)

        self.masked_gauss_q0 = gauss_2d * self.create_mask(self.size)
        self.masked_gauss_q0 = self.masked_gauss_q0 / np.sum(self.masked_gauss_q0)
        
        self.masked_gauss_q1 = np.rot90(self.masked_gauss_q0, -1)
        self.masked_gauss_q2 = np.rot90(self.masked_gauss_q0, -2)
        self.masked_gauss_q3 = np.rot90(self.masked_gauss_q0, -3)

        self.kernel = np.array([self.masked_gauss_q0, self.masked_gauss_q1, 
                           self.masked_gauss_q2, self.masked_gauss_q3])
        
        self.sector_sum = np.sum(self.masked_gauss_q0) #since it's a box, sum of each weights is equal
        

    def create_mask(self,size:int)-> np.ndarray:
        """given a kernel size, create a mask that is 1 for top left quadrant of the matrix
        Args:
            size (int): odd number
        """
        true_size = size // 2 + 1
        mask = np.zeros((size,size))
        for i in range(0,true_size):
            for j in range(0,true_size):
                mask[i, j] = 1
        return mask

    def process_grayscale(self, grayscale: np.ndarray):
        """Given an input image, apply the Kuwahara filter to it.
        All params for the filter are already set in the constructor

        Args:
            input_image (np.ndarray): input image to apply the filter to. Should be a 2D greyscale image
        """
        if np.max(grayscale) <= 1 and grayscale.dtype == np.uint8:
            grayscale = grayscale.astype(np.float64)
            grayscale = grayscale / 255.
        else:
            grayscale = grayscale.astype(np.float64)

        grayscale_squared = grayscale * grayscale

        shape = (self.num_quadrants, grayscale.shape[0], grayscale.shape[1])
        m = np.zeros(shape)
        m_squared = np.zeros(shape)
        weighted_sum_of_squared_image = np.zeros(shape)
        variance = np.zeros(shape)
        sd = np.zeros(shape)
        a = np.zeros(shape)

        weighted_sum_of_quadrants = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        sum_of_quadrant_weights = np.zeros((grayscale.shape[0], grayscale.shape[1]))

        for i in range(self.num_quadrants):
            m[i] = cv2.filter2D(grayscale, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum

            m_squared[i] = cv2.filter2D(grayscale_squared, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            weighted_sum_of_squared_image[i] = m[i] * m[i]
            variance[i] = m_squared[i] - weighted_sum_of_squared_image[i]
            variance[i] = np.maximum(variance[i], 0) # avoid catastrophic cancellation
            sd[i] = np.sqrt(variance[i])
            a[i] = 1/(1 + np.power(np.abs(sd[i]), 8))

            weighted_sum_of_quadrants += a[i] * m[i]    
            sum_of_quadrant_weights += a[i]

        output_image = weighted_sum_of_quadrants / sum_of_quadrant_weights
        output_image = output_image / 255.  #rescale to 0-1 so it shows up :)
        return output_image
    
    def process(self, input_image: np.ndarray):
        """Given an input image, apply the Kuwahara filter to it.
        All params for the filter are already set in the constructor

        Args:
            input_image (np.ndarray): input image to apply the filter to. Should be an RBG image
        """

        grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        image_r = input_image[:,:,0] #TODO: do this but without the split
        image_g = input_image[:,:,1]
        image_b = input_image[:,:,2]

        grayscale = grayscale.astype(np.float64)

        grayscale_squared = grayscale * grayscale

        shape = (self.num_quadrants, grayscale.shape[0], grayscale.shape[1])

        out_r = np.zeros(shape)
        out_g = np.zeros(shape)
        out_b = np.zeros(shape)

        m = np.zeros(shape)
        m_squared = np.zeros(shape)
        weighted_sum_of_squared_image = np.zeros(shape)
        variance = np.zeros(shape)
        sd = np.zeros(shape)
        a = np.zeros(shape)

        weighted_sum_of_quadrants_r = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        weighted_sum_of_quadrants_g = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        weighted_sum_of_quadrants_b = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        sum_of_quadrant_weights = np.zeros((grayscale.shape[0], grayscale.shape[1]))


        for i in range(self.num_quadrants):
            out_r[i] = cv2.filter2D(image_r, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            out_g[i] = cv2.filter2D(image_g, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            out_b[i] = cv2.filter2D(image_b, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            m[i] = cv2.filter2D(grayscale, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum

            m_squared[i] = cv2.filter2D(grayscale_squared, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            weighted_sum_of_squared_image[i] = m[i] * m[i]
            variance[i] = m_squared[i] - weighted_sum_of_squared_image[i]
            variance[i] = np.maximum(variance[i], 0) # avoid catastrophic cancellation
            sd[i] = np.sqrt(variance[i])
            a[i] = 1/(1 + np.power(np.abs(sd[i]), 8))

            weighted_sum_of_quadrants_r += a[i] * out_r[i]
            weighted_sum_of_quadrants_g += a[i] * out_g[i]
            weighted_sum_of_quadrants_b += a[i] * out_b[i]

            sum_of_quadrant_weights += a[i]

        output_image_r = weighted_sum_of_quadrants_r / sum_of_quadrant_weights
        output_image_g = weighted_sum_of_quadrants_g / sum_of_quadrant_weights
        output_image_b = weighted_sum_of_quadrants_b / sum_of_quadrant_weights
        output_image_a = np.ones(output_image_b.shape)*255

        output_image = cv2.merge((output_image_r, output_image_g, output_image_b, output_image_a))

        output_image = output_image / 255.  #rescale to 0-1 so it shows up :)
        return output_image
    
    async def async_process(self, input_image: np.ndarray):
        """Given an input image, apply the Kuwahara filter to it.
        All params for the filter are already set in the constructor

        Args:
            input_image (np.ndarray): input image to apply the filter to. Should be an RBG image
        """

        grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        image_r = input_image[:,:,0] #TODO: do this but without the split
        image_g = input_image[:,:,1]
        image_b = input_image[:,:,2]

        grayscale = grayscale.astype(np.float64)

        grayscale_squared = grayscale * grayscale

        shape = (self.num_quadrants, grayscale.shape[0], grayscale.shape[1])

        await asyncio.sleep(1/120)

        out_r = np.zeros(shape)
        out_g = np.zeros(shape)
        out_b = np.zeros(shape)

        m = np.zeros(shape)
        m_squared = np.zeros(shape)
        weighted_sum_of_squared_image = np.zeros(shape)
        variance = np.zeros(shape)
        sd = np.zeros(shape)
        a = np.zeros(shape)

        await asyncio.sleep(1/120)

        weighted_sum_of_quadrants_r = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        weighted_sum_of_quadrants_g = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        weighted_sum_of_quadrants_b = np.zeros((grayscale.shape[0], grayscale.shape[1]))
        sum_of_quadrant_weights = np.zeros((grayscale.shape[0], grayscale.shape[1]))


        for i in range(self.num_quadrants):
            await asyncio.sleep(1/120)
            out_r[i] = cv2.filter2D(image_r, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            out_g[i] = cv2.filter2D(image_g, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            out_b[i] = cv2.filter2D(image_b, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            m[i] = cv2.filter2D(grayscale, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum

            m_squared[i] = cv2.filter2D(grayscale_squared, -1, self.kernel[i], borderType=cv2.BORDER_DEFAULT) / self.sector_sum
            weighted_sum_of_squared_image[i] = m[i] * m[i]
            variance[i] = m_squared[i] - weighted_sum_of_squared_image[i]
            variance[i] = np.maximum(variance[i], 0) # avoid catastrophic cancellation
            sd[i] = np.sqrt(variance[i])
            a[i] = 1/(1 + np.power(np.abs(sd[i]), 8))

            weighted_sum_of_quadrants_r += a[i] * out_r[i]
            weighted_sum_of_quadrants_g += a[i] * out_g[i]
            weighted_sum_of_quadrants_b += a[i] * out_b[i]

            sum_of_quadrant_weights += a[i]

        await asyncio.sleep(1/120)

        output_image_r = weighted_sum_of_quadrants_r / sum_of_quadrant_weights
        output_image_g = weighted_sum_of_quadrants_g / sum_of_quadrant_weights
        output_image_b = weighted_sum_of_quadrants_b / sum_of_quadrant_weights
        output_image_a = np.ones(output_image_b.shape)*255

        output_image = cv2.merge((output_image_r, output_image_g, output_image_b, output_image_a))

        output_image = output_image / 255.  #rescale to 0-1 so it shows up :)
        return output_image

        