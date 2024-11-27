import numpy as np
import OpenEXR

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
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
        "type" : OpenEXR.scanlineimage }
    channel = OpenEXR.Channel(src.astype('float32'))
    obj = {}
    obj["RGBA"] = channel
    with OpenEXR.File(header, obj) as outfile:
        outfile.write(output_path)
    