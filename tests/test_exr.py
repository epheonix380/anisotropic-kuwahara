import pytest
import OpenEXR
from AnisotropicKuwahara.Generator import RainbowGenerator, BlotchGenerator
from AnisotropicKuwahara.Mirrored import Mirrored
from AnisotropicKuwahara.utils.image_utils import show_image
import matplotlib.pyplot as plt

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
        outfile.write("generator_test.exr")
    # show_image(rainbow_generator.getDst())
    # plt.show()
    assert outfile is not None

def test_blotch_generator(blotch_generator):
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(blotch_generator.getDst().astype('float32'))
    obj = {"RGBA": channel}
    with OpenEXR.File(header, obj) as outfile:
        outfile.write("blotch_generator_test.exr")
    # show_image(rainbow_generator.getDst())
    # plt.show()
    assert outfile is not None

def test_mirrored_filter():
    with OpenEXR.File("generator_test.exr") as infile:
        channels = infile.channels()
        RGBA = channels["RGBA"]
        RGB = infile.channels()["RGBA"].pixels
        mir = Mirrored(RGB)
        mir.singleProcessAll()
        channels["RGBA"] = OpenEXR.Channel(mir.getDst().astype('float32'))
        infile.write("mirrored_test.exr")
    # show_image(rainbow_generator.getDst())
    # plt.show()
    assert infile is not None