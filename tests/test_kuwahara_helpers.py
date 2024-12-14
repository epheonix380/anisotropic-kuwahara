import pytest
import matplotlib.pyplot as plt
from pykuwahara import kuwahara as pykuwahara
import time
import cv2
import numpy as np

from AnisotropicKuwahara.utils.ImageUtil import *
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


def test_kuwahara_kernel_size():
    image = read_image("examples/shrek.jpg")
    # image = downsample_image(image, image.shape[0]//50)

    # kernels = [5,10,20,40,80,160]
    repeats = 3
    kernels = [5,10,20,40,80,160]
    times = []
    time_report = TimeReport("")
    for kernel in kernels:
        print(f"kernel size: {kernel}")
        start = time.time()
        k = Kuwahara(kernel_radius=kernel)
        for x in range(repeats):
                result = k.process(image)

        difference = time.time() - start
        
        times.append(difference/repeats)

    plt.plot(kernels, times)
    plt.xlabel("Kernel size")
    plt.ylabel("Time (s)")
    plt.title("Kuwahara runtime vs kernel size")
    # plt.show()



def test_multiple_convolutions():
    image = read_image("examples/shrek.jpg")
    # image = downsample_image(image, image.shape[0]//50)

    repeats = 3

    kernel_size = 4



@pytest.mark.parametrize("kernel", [5,10,20,40,80,100])
def test_kuwahara_init(kernel):
    image = read_image("examples/shrek.jpg")
    # image = downsample_image(image, image.shape[0]//50)
    image_r = image[:,:,0]
    image_g = image[:,:,1]
    image_b = image[:,:,2]

    print(kernel)
    k = Kuwahara(kernel_radius=kernel)

    with TimeReport(f"doing kuwahara with kernel size {kernel}"):
        result_r = k.process_grayscale(image_r)
        result_g = k.process_grayscale(image_g)
        result_b = k.process_grayscale(image_b)
        result_rgb = cv2.merge([result_r, result_g, result_b])
    with TimeReport(f"only grayscale, with kernel size {kernel}"):
        result_from_grayscale = k.process(image, dtype=np.float32)
    with TimeReport(f"import, with kernel size {kernel}"):
        result_grayscale = pykuwahara(image, method="gaussian", radius=kernel)
    with TimeReport(f"hsv, with kernel size {kernel}"):
        result_hsv = k.process_hsv(image)

    plot_and_save(
        [result_rgb, result_from_grayscale,result_hsv,result_grayscale], 
        titles=["original","grayscale","hsv", "external implementation"], 
        save_path=f"tests/output/color_grayscale_diff/kuwahara_kernel_{kernel}.png")



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