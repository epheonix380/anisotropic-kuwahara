import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to an image. This is a low-pass filter that removes high-frequency noise from the image.
    Args:
        image (np.ndarray): image to apply filter to
        kernel_size (int): size of the kernel. Must be odd
        sigma (float): standard deviation of the Gaussian distribution
    Returns:
        np.ndarray: filtered image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def read_image(image_path: str) -> np.ndarray:
    """Read image from file path. open cv reads in BGR so this converts it to RGBA format.

    Args:
        image_path (str): path to image file

    Returns:
        np.ndarray: image as ndarray. Shape is (height, width, 4) where 4 is the RGBA channels. Values are in the range 0-255
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img


def crop_image(image: np.ndarray,  width: int, height: int, x: int = 0, y: int = 0) -> np.ndarray:
    """Crop image to given dimensions, with the top left corner at x, y
    """
    return image[y:y+height, x:x+width]


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample image by factor. Image is resized to 1/factor of the original size.
    """
    if factor < 1:
        raise ValueError("Factor must be greater than 1")

    # usually when we downsample we want to use a low-pass filter (Eg. Gaussian) to avoid aliasing
    # but apparently Inter_Area prevents anti-aliasing so we dont need to use a low-pass filter
    # https://pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
    # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
    return cv2.resize(image, (image.shape[1]//factor, image.shape[0]//factor), interpolation=cv2.INTER_AREA)


def plot_orientation_arrows(orientations, image, title_text="title"):
    """Given orientation angle and the original image draw the orientation arrow on top of the image
        must call plt.plot() to see the plots.Expects both orientation and image to have the same with and height
    Args:
        orientations np.ndarray of the arrow angles
        image np.ndarray: image array
        title_text (str, optional): title of plot
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title_text)

    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            angle = orientations[i, j]
            x = j
            y = i
            dx = np.cos(angle)
            dy = np.sin(angle)
            plt.arrow(x, y, dx, dy, head_width=0.1,
                      head_length=0.2, fc='red', ec='red')


def plot_image(data: np.ndarray, show_values=False, cv_color_transform=None, channel_to_show=0, title=None, cmap=None) -> None:
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

                plt.text(x, y, f'{display_value:.2f}', ha='center',
                         va='center', color='red', fontsize=6)


def plot_and_save(images, orientations=None, titles=None, save_path="output.png", show_values=False, cv_color_transform=None, cmap=None):
    """
    Creates a grid of subplots for the given images and (optional) orientations, 
    saves the entire plot as a single image file.

    Args:
        images (list[np.ndarray]): List of image arrays to plot.
        orientations (list[np.ndarray], optional): List of orientation arrays corresponding to each image. Defaults to None.
        titles (list[str], optional): List of titles for each subplot. Defaults to None.
        save_path (str, optional): File path to save the resulting image. Defaults to "output.png".
        show_values (bool, optional): Whether to show pixel values on the images. Defaults to False.
        cv_color_transform (int, optional): OpenCV color transformation constant. Defaults to None.
        cmap (str, optional): Matplotlib colormap to use (e.g., 'gray'). Defaults to None.

    Raises:
        ValueError: If the `images` list is empty.
    """
    if not images or len(images) == 0:
        raise ValueError("The `images` list cannot be empty.")

    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1) # flatten for speed
    
    for idx, ax in enumerate(axes):
        if idx < num_images:
            image = images[idx]
            title = titles[idx] if titles and idx < len(titles) else None
            orientation = orientations[idx] if orientations and idx < len(orientations) else None

            if cv_color_transform is not None:
                ax.imshow(cv2.cvtColor(image, cv_color_transform), cmap=cmap)
            else:
                ax.imshow(image, cmap=cmap)

            if title:
                ax.set_title(title)

            if orientation is not None:
                for i in range(orientation.shape[0]):
                    for j in range(orientation.shape[1]):
                        angle = orientation[i, j]
                        x = j
                        y = i
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='red', ec='red')
            
            if show_values:
                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        pixel_val = image[y, x]
                        display_value = pixel_val if len(image.shape) == 2 else pixel_val[0]
                        ax.text(x, y, f'{display_value:.2f}', ha='center', va='center', color='red', fontsize=6)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)