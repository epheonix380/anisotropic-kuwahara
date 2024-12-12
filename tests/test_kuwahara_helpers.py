import pytest
import matplotlib.pyplot as plt
from pykuwahara import kuwahara as pykuwahara

import cv2
import numpy as np

from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian, crop_image
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.Kuwahara import Kuwahara
from AnisotropicKuwahara.utils.TimeReport import TimeReport

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
cv imread reads in 0-255 and in BGR format, so the image is converted to RGBA format on the fixture
"""

@pytest.fixture
def time_report(request):
    return TimeReport(request.node.name)


def create_mask(size):
    true_size = size // 2 + 1
    mask = np.zeros((size,size))
    for i in range(0,true_size):
        for j in range(0,true_size):
            mask[i, j] = 1
    return mask

def addPadding(image, pad):
    return np.pad(image, ((pad,pad), (pad,pad)), mode='constant', constant_values=0)


def test_kuwahara_init():
    image = read_image("examples/shrek.jpg")
    # image = downsample_image(image, image.shape[0]//50)
    image_r = image[:,:,0]
    image_g = image[:,:,1]
    image_b = image[:,:,2]

    with TimeReport("kuwahara init"):
        k = Kuwahara(kernel_radius=11)

    with TimeReport("doing kuwahara"):
        result_r = k.process_grayscale(image_r)
        result_g = k.process_grayscale(image_g)
        result_b = k.process_grayscale(image_b)
        result = cv2.merge([result_r, result_g, result_b])

    with TimeReport("only grayscale"):
        result_grayscale = k.process(image)




@pytest.mark.parametrize("width", [100,1000,2000])
@pytest.mark.parametrize("power", [1,4,8, 16])
def test_squaring_large_matrixes(width, power):
    test_matrix = np.random.rand(width, width // 2)
    tries = 100

    with TimeReport(f"power of {power} large matrix of size {width} * {width/2} using numpy"):
        for x in range(tries):
            test_matrix_squared = np.power(test_matrix, power)
    with TimeReport(f"power of {power} large matrix of size {width} * {width/2} using for **"):
        for x in range(tries):
            test_matrix_squared = test_matrix ** power

    print("=====")