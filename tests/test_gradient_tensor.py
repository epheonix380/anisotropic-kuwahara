import pytest
from AnisotropicKuwahara.Kuwahara import Kuwahara
from AnisotropicKuwahara.AnisotropicKuwahara import KuwaharaAnisotropic

from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.utils.TimeReport import TimeReport
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

"""
Tests for kuwahara runtime :)
if print statements aren't working a workaround is to use warnings to print the message (yep, it's kinda jank)
cv imread reads in 0-255 and in BGR format, so the image is converted to RGBA format on the fixture
"""

@pytest.fixture
def time_report(request):
    return TimeReport(request.node.name)


def test_for_loop_tensor_size_1900(time_report):
    img = read_image("examples/shapes_large.jpg")
    plot_image(img, title="Original Image")

    with time_report:
        st = StructuredTensor(img)

        # process the tensor
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                st.process([x,y])

        dxdx, dydy, dxdy = [st.getDst()[:,:,i] for i in range(3)]

        plot_image(dxdx, title="dxdx", cmap='gray', show_values=True)
        plot_image(dydy, title="dydy",cmap='gray', show_values=True)
        plot_image(dxdy, title="dxdy", cmap='gray', show_values=True)

    # plt.show() # uncomment to show the images. This will open a new window with the images and stall tests tho...
    assert time_report.runtime < 100.0


# TODO: check how these following CV data types affect output when used on convolution and how it affects everything else
# def CV_8UC(channels: int) -> int: ...
# def CV_8SC(channels: int) -> int: ...
# def CV_16UC(channels: int) -> int: ...
# def CV_16SC(channels: int) -> int: ...
# def CV_32SC(channels: int) -> int: ...
# def CV_32FC(channels: int) -> int: ...
# def CV_64FC(channels: int) -> int: ...
# def CV_16FC(channels: int) -> int: ...
@pytest.mark.parametrize("image_dtype", [np.uint8, np.uint16, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("out_dtype", [-1]) #-1 means it will follow original image dtype
def test_rgb_tensor_different_dtypes_no_errors(image_dtype, out_dtype, time_report): #TODO: complete this test
    with time_report:
        img = read_image("examples/shrek.jpg")
        original_dtype = img.dtype
        img = gaussian(img, 11, 2.0)
        # img = np.array(img, dtype=image_dtype)

        if image_dtype == np.float32 or image_dtype == np.float64:
            img = img / 255.0

        # plot_image(img, title="Original Image")

        st = StructuredTensor(img)
        conv_tensor = st.getGradientsRGB(out_dtype)
        dxdx_conv, dydy_conv, dxdy_conv = [conv_tensor[:,:,i] for i in range(3)]

        # plot_image(dxdx_conv, title="dxdx conv", cmap='gray')
        # plot_image(dydy_conv, title="dydy conv", cmap='gray')
        # plot_image(dxdy_conv, title="dxdy conv", cmap='gray')
        # plt.show()

        # assert conv_tensor.dtype == original_dtype
        assert conv_tensor.shape == (img.shape[0], img.shape[1], 3)
    assert time_report.runtime < 10.0

@pytest.mark.parametrize("kernel_size", [1,3,11,25,51,101])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0,4.0,8])
def test_with_gauss_tensor_RGB_2D_conv__no_downsample(time_report, kernel_size, sigma):
    img = read_image("examples/shrek.jpg") #TODO: replace shrek image with something less scary ;-;
    img = gaussian(img, kernel_size, sigma)

    st = StructuredTensor(img)

    with time_report:
        conv_tensor = st.getGradientsRGB(cv2.CV_64F)
        dxdx_conv, dydy_conv, dxdy_conv = [conv_tensor[:,:,i] for i in range(3)]
        dxdx_conv = cv2.normalize(dxdx_conv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        cv2.imwrite(f"tests/output/test_kuwahara_runtime/RGB_tensor_2D_conv_no_downsample_{kernel_size}_{sigma}.png", cv2.cvtColor(dxdx_conv, cv2.COLOR_GRAY2BGRA))

    assert time_report.runtime < 10.0

def test_compare_tensor_creation__with_2D_conv_300_wide_downsampled(time_report):
    img = read_image("examples/shrek.jpg")
    downsampled_RGBA = downsample_image(img, img.shape[1]//300)
    downsampled_RGBA = img
    # plot_image(downsampled_RGBA, title="Original Image, downsampled") # uncomment to see the image after downsampling
    
    st = StructuredTensor(downsampled_RGBA)

    #original implementation
    with time_report:
        height, width, _ = downsampled_RGBA.shape
        for y in range(height):
            for x in range(width):
                st.process([x,y])
        dxdx, dydy, dxdy = [st.getDst()[:,:,i] for i in range(3)]
    # rgb implementation
    with TimeReport("RGB 2D implementation"):
        conv_tensor = st.getGradientsRGB(cv2.CV_64F)
        dxdx_conv, dydy_conv, dxdy_conv = [conv_tensor[:,:,i] for i in range(3)]
    # grayscale implementation
    with TimeReport("greyscale 2D implementation"):
        conv_tensor_grey = st.getGradientGray()
        dxdx_conv_grey, dydy_conv_grey, dxdy_conv_grey = [conv_tensor_grey[:,:,i] for i in range(3)]
    with TimeReport("greyscale 1D implementation"):
        conv_tensor_grey_1d = st.getGradientGraySeparated()
        dxdx_conv_grey_1d, dydy_conv_grey_1d, dxdy_conv_grey_1d = [conv_tensor_grey_1d[:,:,i] for i in range(3)]
        
    # plt.show()

def test_compare_grayscale_2D_conv_with_grayscale_1d_conv(time_report):
    img = read_image("examples/shrek.jpg")
    downsampled_RGBA = img
    downsampled_RGBA = gaussian(downsampled_RGBA, 11, 2.0)
    # plot_image(downsampled_RGBA, title="Original Image, downsampled") # uncomment to see the image after downsampling
    
    st = StructuredTensor(downsampled_RGBA)

    # grayscale implementation
    with TimeReport("greyscale 2D implementation"):
        conv_tensor_grey_2d = st.getGradientGray()
        dxdx_conv_grey_2d, dydy_conv_grey_2d, dxdy_conv_grey_2d = [conv_tensor_grey_2d[:,:,i] for i in range(3)]


        # grayscale implementation
    with TimeReport("greyscale 1D implementation"):
        conv_tensor_grey_1d = st.getGradientGraySeparated()
        dxdx_conv_grey_1d, dydy_conv_grey_1d, dxdy_conv_grey_1d = [conv_tensor_grey_1d[:,:,i] for i in range(3)]

    assert np.allclose(dxdx_conv_grey_2d, dxdx_conv_grey_1d, atol=1e-2)
    assert np.allclose(dydy_conv_grey_2d, dydy_conv_grey_1d, atol=1e-2)
    assert np.allclose(dxdy_conv_grey_2d, dxdy_conv_grey_1d, atol=1e-2)
 