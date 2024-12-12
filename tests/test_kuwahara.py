import pytest
import matplotlib.pyplot as plt

import cv2
import numpy as np

from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian, crop_image
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.AnisotropicKuwahara import KuwaharaAnisotropic
from AnisotropicKuwahara.utils.TimeReport import TimeReport

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
cv imread reads in 0-255 and in BGR format, so the image is converted to RGBA format on the fixture
"""

@pytest.fixture
def time_report(request):
    return TimeReport(request.node.name)


def test_original_kuwahara_runtime_small(time_report):
    img = read_image("examples/shrek.jpg")
    # img = cv2.convertScaleAbs(img)
    # img = crop_image(img, 200,200,475, 75) # crop the image to a smaller size
    # img = downsample_image(img, img.shape[0]//30)
    # plot_image(img, title="downsampled Image")
    

    st = StructuredTensor(img)
    structure_tensor = st.getGradientGray()

    ka = KuwaharaAnisotropic(structure_tensor=structure_tensor, src=img, size=5)
    with TimeReport("using conv and numpy"):
        ka.get_results(structure_tensor)


        # plot_image(output, title="Kuwahara Anisotropic Image", cmap='gray')
    plt.show() # uncomment to show the images. This will open a new window with the images and stall tests tho...
