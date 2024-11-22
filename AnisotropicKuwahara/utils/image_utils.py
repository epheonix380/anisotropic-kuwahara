import numpy as np
import cv2
import matplotlib.pyplot as plt


def downsample_image(image:np.ndarray, factor:int):
    """Downsample image by factor. Image is resized to 1/factor of the original size.
    """
    return cv2.resize(image, (image.shape[1]//factor, image.shape[0]//factor), interpolation=cv2.INTER_LINEAR)


def plot_image(data:np.ndarray, show_values=False, cv_color_transform=None):
    """Given nd array image, plot the image in a new window. Must call plt.show() to display the image.
    Note that the image needs to be in RGB format to show properly. 
    Multiple windows can be opened at once by calling this function multiple times before calling plt.show()
    Args:
        data (np.ndarray): image as ndarray
        show_values (bool, optional): _description_. Defaults to False.
        cv_color_transform: cv2 color constant. Defaults to None.
    """
    plt.figure()
    if cv_color_transform is not None:
        plt.imshow(cv2.cvtColor(data, cv_color_transform))
    else:
        plt.imshow(data)

    if show_values:
        for (i, j), val in np.ndenumerate(data):
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='red')




