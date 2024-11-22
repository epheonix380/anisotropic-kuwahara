from .Filter import Filter
import sys
import numpy as np
import OpenEXR

def filterWithMultiprocessing(input_path:str, output_path:str, filter: Filter = None):
""" Applies filter to EXR found on input_path and saves it on the output_path. Uses Multithreading
"""
    with OpenEXR.File(input) as infile:
        channels = infile.channels()
        RGB = infile.channels()["RGBA"].pixels
        filter.setSrc(src=RGB)
        filter.multiProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
        channels["RGBA"] = OpenEXR.Channel(filter.getDst().astype('float32'))
        infile.write(output)

def filter(input, output, filter: Filter = None):
    with OpenEXR.File(input) as infile:
        channels = infile.channels()
        RGB = infile.channels()["RGBA"].pixels
        filter.setSrc(src=RGB)
        filter.singleProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
        channels["RGBA"] = OpenEXR.Channel(filter.getDst().astype('float32'))
        infile.write(output)

def read(input):
    with OpenEXR.File(input) as infile:
        channels = infile.channels()
        return infile.channels()["RGBA"].pixels
    
def write(output, src):
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
        "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(src.astype('float32'))
    obj = {}
    obj["RGBA"] = channel
    with OpenEXR.File(header, obj) as outfile:
        outfile.write(output)
    