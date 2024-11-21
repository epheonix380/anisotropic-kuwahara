from .AnisotropicKuwahara import KuwaharaAnisotropic
from .Tensor import StructuredTensor
from .EXR import read, write
import numpy as np

def main(input, output):
    RGBA = read(input=input)
    print(RGBA.shape)
    height, width, _ = RGBA.shape
    st = StructuredTensor(RGBA)
    for y in range(height):
        for x in range(width):
            st.process([x,y])
    ka = KuwaharaAnisotropic(structure_tensor=st.getDst(), src=RGBA)
    ka.multiProcessAll() # pipenv shell python -m AnisotropicKuwahara BG_depthInfo.exr test.exr
    write(output=output, src=ka.getDst())
  