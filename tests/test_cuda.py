import pytest
import matplotlib.pyplot as plt
import cv2
import numpy as np

from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.utils.TimeReport import TimeReport
from AnisotropicKuwahara.AnisotropicKuwahara import KuwaharaAnisotropic
from AnisotropicKuwahara.KuwaCuda import kuwaCuda

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
cv imread reads in 0-255 and in BGR format, so the image is converted to RGBA format on the fixture
"""

@pytest.fixture
def time_report(request):
    return TimeReport(request.node.name)

@pytest.fixture
def structured_tensor(request):
    img = read_image("examples/shapes_all_colors.jpg")
    downsampled_RGBA = img
    # plot_image(downsampled_RGBA, title="Original Image, downsampled") # uncomment to see the image after downsampling
    print("Starting Tensor")
    st = StructuredTensor(downsampled_RGBA)
    height, width, _ = downsampled_RGBA.shape
    for y in range(height):
        for x in range(width):
            st.process([x,y])
    return st.getDst()

def test_compare_kuwahara_multithreaded_with_2D_conv_300_wide_downsampled(structured_tensor):
    img = read_image("examples/shapes_all_colors.jpg")
    downsampled_RGBA = img
    #original implementation
    print("Tensor finished")
    akMulti = KuwaharaAnisotropic(structure_tensor=structured_tensor, src=downsampled_RGBA)
    print("Starting multi")
    with TimeReport("Multi Processing"):
        akMulti.multiProcessAll()
        print("Multi finished")

def test_compare_kuwahara_cuda_with_2D_conv_300_wide_downsampled(structured_tensor):
    img = read_image("examples/shapes_all_colors.jpg")
    downsampled_RGBA = img
    print("Starting multi")
    with TimeReport("Multi Processing"):
        kuwaCuda(img, structured_tensor=structured_tensor)
        print("Multi finished")

