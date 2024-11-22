import pytest
import OpenEXR
from AnisotropicKuwahara.Kuwahara import Kuwahara
from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image
from AnisotropicKuwahara.utils.TimeReport import TimeReport
import matplotlib.pyplot as plt
import cv2

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
"""
SHOW_IMAGES = True

@pytest.fixture
def time_report(request):
    return TimeReport(request.node.name)

@pytest.fixture
def sample_image():
    image_path = "examples/shrek.jpg"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = img / 255.0
    return img

def test_10_wide_image(sample_image,time_report):
    downsampled = downsample_image(sample_image, sample_image.shape[1]//10)
    plot_image(downsampled)
    with time_report:
        mir = Kuwahara(downsampled, src=downsampled)
        mir.singleProcessAll()
        plot_image(mir.getDst(), show_values=True, channel_to_show=0)

    if SHOW_IMAGES:
        plt.show()
    assert time_report.runtime < 1.0

def test_100_wide_image(sample_image,time_report):
    downsampled = downsample_image(sample_image, sample_image.shape[1]//100)
    with time_report:
        mir = Kuwahara(downsampled, src=downsampled)
        mir.singleProcessAll()
        plot_image(mir.getDst())

    if SHOW_IMAGES:
        plt.show()
    assert time_report.runtime < 1.0


def test_400_wide_image(sample_image,time_report):
    downsampled = downsample_image(sample_image, sample_image.shape[1]//400)

    with time_report:
        mir = Kuwahara(downsampled, src=downsampled)
        mir.singleProcessAll()
        plot_image(mir.getDst())

    if SHOW_IMAGES:
        plt.show()
    assert time_report.runtime < 1.0