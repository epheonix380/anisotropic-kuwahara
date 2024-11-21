import OpenEXR
from .Generator import RainbowGenerator, BlotchGenerator
from .Mirrored import Mirrored

def test():
    print("Running Rainbow Image Generator Test")
    generator = RainbowGenerator()
    generator.singleProcessAll()
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
            "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(generator.getDst().astype('float32'))
    obj = {}
    obj["RGBA"] = channel
    with OpenEXR.File(header, obj) as outfile:
        outfile.write("generator_test.exr")
    print("Running Image Generator Test")
    generator2 = BlotchGenerator()
    generator2.singleProcessAll()
    header2 = { "compression" : OpenEXR.ZIP_COMPRESSION,
            "type" : OpenEXR.scanlineimage }
    channel2 = OpenEXR.Channel(generator2.getDst().astype('float32'))
    obj2 = {}
    obj2["RGBA"] = channel2
    with OpenEXR.File(header2, obj2) as outfile2:
        outfile2.write("blotch_generator_test.exr")
    print("Finished Image Generator Test, Running Mirror Filter Test")
    with OpenEXR.File("generator_test.exr") as infile:
        channels = infile.channels()
        RGBA = channels["RGBA"]
        RGB = infile.channels()["RGBA"].pixels
        mir = Mirrored(RGB)
        mir.singleProcessAll()
        channels["RGBA"] = OpenEXR.Channel(mir.getDst().astype('float32'))
        infile.write("mirrored_test.exr")
    print("Finished Mirro Filter Test")