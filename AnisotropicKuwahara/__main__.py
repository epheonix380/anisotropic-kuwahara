from .AnisotropicKuwahara import KuwaharaAnisotropic
from .Tensor import StructuredTensor
import sys
import numpy as np
import OpenEXR

def main():
    arguements = sys.argv[1:]
    if len(arguements) >= 2:
        input = arguements[0]
        output = arguements[1]
        with OpenEXR.File(input) as infile:
            channels = infile.channels()
            RGBA = channels["RGBA"]
            print(RGBA)
            RGB = infile.channels()["RGBA"].pixels
            print(RGB.shape)
            print(channels)
            height, width, _ = RGB.shape
            st = StructuredTensor(RGB)
            for y in range(height):
                for x in range(width):
                    st.process([x,y])
            ka = KuwaharaAnisotropic(RGB, st.getDst())
            ka.multiProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
            channels["RGBA"] = OpenEXR.Channel(ka.getDst().astype('float32'))
            infile.write(output)
    else:
        print("Wrong Arguements")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()