import pytest
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

from AnisotropicKuwahara.utils.ImageUtil import plot_and_save

@pytest.fixture
def sample_image():
    # Grayscale random image
    return np.random.rand(10, 10)

@pytest.fixture
def sample_color_image():
    # RGB random image
    return (np.random.rand(10, 10, 3) * 255).astype(np.uint8)

@pytest.fixture
def sample_orientation():
    # Random orientations in radians
    return np.random.rand(10, 10) * 2 * np.pi

def test_single_image_no_orientation(sample_image):
    output_file = "test_single_no_orientation.png"
    plot_and_save(images=[sample_image], save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_multiple_images_no_orientation(sample_image):
    output_file = "test_multiple_no_orientation.png"
    plot_and_save(images=[sample_image, sample_image], titles=["Image 1", "Image 2"], save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_single_image_with_orientation(sample_image, sample_orientation):
    output_file = "test_single_with_orientation.png"
    plot_and_save(images=[sample_image], orientations=[sample_orientation], save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_multiple_images_with_orientation(sample_image, sample_orientation):
    output_file = "test_multiple_with_orientation.png"
    plot_and_save(
        images=[sample_image, sample_image],
        orientations=[sample_orientation, sample_orientation],
        titles=["Image 1", "Image 2"],
        save_path=output_file,
        cmap='gray',
    )
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_missing_titles(sample_image):
    output_file = "test_missing_titles.png"
    plot_and_save(images=[sample_image, sample_image], save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_empty_inputs():
    output_file = "test_empty_inputs.png"
    with pytest.raises(ValueError, match="The `images` list cannot be empty."):
        plot_and_save(images=[], save_path=output_file)

def test_color_transform(sample_color_image):
    output_file = "test_color_transform.png"
    plot_and_save(images=[sample_color_image], save_path=output_file, cv_color_transform=cv2.COLOR_BGR2RGB)
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_pixel_value_annotations(sample_image):
    output_file = "test_pixel_values.png"
    plot_and_save(images=[sample_image], save_path=output_file, cmap='gray', show_values=True)
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_large_number_of_images():
    output_file = "test_large_number_of_images.png"
    num_images = 99
    images = [np.random.rand(10, 10) for _ in range(num_images)]
    titles = [f"Image {i+1}" for i in range(num_images)]

    plot_and_save(images=images, titles=titles, save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)

def test_mixed_orientations():
    output_file = "test_mixed_orientations.png"
    num_images = 10
    half_with_orientations = num_images // 2

    images = [np.random.rand(10, 10) for _ in range(num_images)]

    # Assign orientations to the first half and None to the second half
    orientations = [np.random.rand(10, 10) * 2 * np.pi if i < half_with_orientations else None for i in range(num_images)]
    titles = [f"Image {i+1}" for i in range(num_images)]

    plot_and_save(images=images, orientations=orientations, titles=titles, save_path=output_file, cmap='gray')
    assert os.path.exists(output_file)
    os.remove(output_file)
