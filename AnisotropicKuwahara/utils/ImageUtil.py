import numpy as np
import cv2
import matplotlib.pyplot as plt

def crop_image(image:np.ndarray,  width:int, height:int, x:int = 0, y:int = 0) -> np.ndarray:
    """Crop image to given dimensions, with the top left corner at x, y
    """
    return image[y:y+height, x:x+width]

def downsample_image(image:np.ndarray, factor:int) -> np.ndarray:
    """Downsample image by factor. Image is resized to 1/factor of the original size.
    """
    if factor < 1:
        raise ValueError("Factor must be greater than 1")
    
    return cv2.resize(image, (image.shape[1]//factor, image.shape[0]//factor), interpolation=cv2.INTER_LINEAR)


def plot_image(data:np.ndarray, show_values=False, cv_color_transform=None, channel_to_show=0) -> None:
    """Given nd array image, plot the image in a new window. Must call plt.show() to display the image.
    Note that the image needs to be in RGB format to show properly. 
    Multiple windows can be opened at once by calling this function multiple times before calling plt.show()
    Args:
        data (np.ndarray): image as ndarray
        show_values (bool, optional): whether to show the channel color . Defaults to False.
        cv_color_transform: cv2 color constant. Defaults to None.
        channel_to_show: channel to show the values of. Defaults to 0 (aka first channel)
    """
    plt.figure()
    if cv_color_transform is not None:
        plt.imshow(cv2.cvtColor(data, cv_color_transform))
    else:
        plt.imshow(data)

    if show_values:
        print(data)
        data = np.round(data, 2)
        for index, val in np.ndenumerate(data[:,:,0]): 
            # prints the first channel of the image
            plt.text(index[1], index[0], f'{data[index[0], index[1], channel_to_show]}', ha='center', va='center', color='red', fontsize=6)




