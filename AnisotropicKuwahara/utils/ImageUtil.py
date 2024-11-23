import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian(image:np.ndarray, kernel_size:int, sigma:float) -> np.ndarray:
    """Apply a Gaussian filter to an image. This is a low-pass filter that removes high-frequency noise from the image.
    Args:
        image (np.ndarray): image to apply filter to
        kernel_size (int): size of the kernel. Must be odd
        sigma (float): standard deviation of the Gaussian distribution
    Returns:
        np.ndarray: filtered image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def read_image(image_path:str) -> np.ndarray:
    """Read image from file path. open cv reads in BGR so this converts it to RGBA format.

    Args:
        image_path (str): path to image file

    Returns:
        np.ndarray: image as ndarray. Shape is (height, width, 4) where 4 is the RGBA channels. Values are in the range 0-255
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def crop_image(image:np.ndarray,  width:int, height:int, x:int = 0, y:int = 0) -> np.ndarray:
    """Crop image to given dimensions, with the top left corner at x, y
    """
    return image[y:y+height, x:x+width]

def downsample_image(image:np.ndarray, factor:int) -> np.ndarray:
    """Downsample image by factor. Image is resized to 1/factor of the original size.
    """
    if factor < 1:
        raise ValueError("Factor must be greater than 1")
    
    # usually when we downsample we want to use a low-pass filter (Eg. Gaussian) to avoid aliasing
    # but apparently Inter_Area prevents anti-aliasing so we dont need to use a low-pass filter
    # https://pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
    # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3         
    return cv2.resize(image, (image.shape[1]//factor, image.shape[0]//factor), interpolation=cv2.INTER_AREA)


def plot_image(data:np.ndarray, show_values=False, cv_color_transform=None, channel_to_show=0, title=None, cmap=None) -> None:
    """Given nd array image, plot the image in a new window. Must call plt.show() to display the image.
    Note that the image needs to be in RGB format to show properly. 
    Multiple windows can be opened at once by calling this function multiple times before calling plt.show()
    Args:
        data (np.ndarray): image as ndarray
        show_values (bool, optional): whether to show the values of a channel. Defaults to False.
        cmap: matplotlib colormap to use. Can be `gray` for grayscale. 
        channel_to_show: channel to show the values of. Defaults to 0 (aka first channel)
        cv_color_transform: cv2 color constant. Defaults to None.
    """
    plt.figure()
    if cv_color_transform is not None:
        plt.imshow(cv2.cvtColor(data, cv_color_transform), cmap=cmap)
    else:
        plt.imshow(data, cmap=cmap)

    if title is not None:
        plt.title(title)

    data_shape_len = len(data.shape)
    if show_values:
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                pixel_val = data[y, x]
                display_value = pixel_val if data_shape_len == 2 else pixel_val[channel_to_show]

                plt.text(x, y, f'{display_value:.2f}', ha='center', va='center', color='red', fontsize=6)






