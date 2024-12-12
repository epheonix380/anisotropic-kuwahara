from .Filter import Filter
from typing import Union
import sys
import numpy as np
import OpenEXR

def filterWithMultiprocessing(input_path:str, output_path:str, filter: Filter = None) -> None:
    """ Applies filter to EXR found on input_path and saves it on the output_path. 
        Uses Multithreading
    """
    with OpenEXR.File(input_path) as infile:
        channels = infile.channels()
        RGB = infile.channels()["RGBA"].pixels
        filter.setSrc(src=RGB)
        filter.multiProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
        channels["RGBA"] = OpenEXR.Channel(filter.getDst().astype('float32'))
        infile.write(output_path)

def filter(input_path: str, output_path:str, filter: Filter = None) -> None: 
    """ Applies filter to EXR found on input file and saves it on the output path
    """
    with OpenEXR.File(input_path) as infile:
        channels = infile.channels()
        RGB = infile.channels()["RGBA"].pixels
        filter.setSrc(src=RGB)
        filter.singleProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
        channels["RGBA"] = OpenEXR.Channel(filter.getDst().astype('float32'))
        infile.write(output_path)

def read(input_path: str) -> np.ndarray:
    """
    Reads a given pathlike string as an exr image and returns it as a ndarray
    """
    with OpenEXR.File(input_path) as infile:
        pixels = infile.channels()["RGBA"].pixels
        print(pixels.shape)
        return pixels
    
def write(output_path:str, src:np.ndarray) -> None:
    """
    Writes given ndarray as an exr, defaults to scanlineimage type.
    """
    path = "pythonprog"
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
        "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(src.astype('float32'))
    obj = {}
    obj["RGBA"] = channel
    with OpenEXR.File(header, obj) as outfile:
        outfile.write(output_path)
    