import pytest
import OpenEXR
from AnisotropicKuwahara.Generator import RainbowGenerator, BlotchGenerator
from AnisotropicKuwahara.Mirrored import Mirrored
from AnisotropicKuwahara.utils.ImageUtil import plot_image
import matplotlib.pyplot as plt
"""
Tests for EXR file generation and processing
"""
SHOW_IMAGES = False

@pytest.fixture
def rainbow_generator():
    generator = RainbowGenerator()
    generator.singleProcessAll()
    return generator

@pytest.fixture
def blotch_generator():
    generator = BlotchGenerator()
    generator.singleProcessAll()
    return generator

def test_rainbow_generator(rainbow_generator):
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(rainbow_generator.getDst().astype('float32'))
    obj = {"RGBA": channel}
    with OpenEXR.File(header, obj) as outfile:
        outfile.write("tests/output/generator_test.exr")
    if SHOW_IMAGES:
        plot_image(rainbow_generator.getDst())
        plt.show()
    assert outfile is not None

def test_blotch_generator(blotch_generator):
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(blotch_generator.getDst().astype('float32'))
    obj = {"RGBA": channel}
    with OpenEXR.File(header, obj) as outfile:
        outfile.write("tests/output/blotch_generator_test.exr")
    if SHOW_IMAGES:
        plot_image(blotch_generator.getDst())
        plt.show()
    assert outfile is not None

def test_mirrored_filter():
    with OpenEXR.File("tests/output/generator_test.exr") as infile:
        channels = infile.channels()
        RGBA = channels["RGBA"]
        RGB = infile.channels()["RGBA"].pixels
        mir = Mirrored(RGB)
        mir.singleProcessAll()
        channels["RGBA"] = OpenEXR.Channel(mir.getDst().astype('float32'))
        infile.write("tests/output/mirrored_test.exr")
    if SHOW_IMAGES:
        plot_image(mir.getDst())
        plt.show()
    assert infile is not None