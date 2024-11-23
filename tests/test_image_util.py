import pytest

from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian
from AnisotropicKuwahara.utils.TimeReport import TimeReport
import matplotlib.pyplot as plt
import cv2
import numpy as np

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
cv imread reads in 0-255 and in BGR format, so the image is converted to RGBA format on the fixture
"""

# @pytest.fixture
# def time_report(request):
#     return TimeReport(request.node.name)


@pytest.mark.parametrize("kernel_size", [3, 5,11,25,51,101,201])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0,4.0,16])
def test_gaussian_filter(kernel_size, sigma):
    img = read_image("examples/shrek.jpg")
    time_report = TimeReport(f"Gaussian Filter {kernel_size} {sigma}")
    with time_report:
        filtered = gaussian(img, kernel_size, sigma)
        # save image 
        cv2.imwrite(f"tests/output/test_image_util_gaussian/shrek_gaussian_{kernel_size}_{sigma}.png", cv2.cvtColor(filtered, cv2.COLOR_RGBA2BGRA))

