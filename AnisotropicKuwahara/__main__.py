from .AnisotropicKuwahara import KuwaharaAnisotropic
from .Tensor import StructuredTensor
import sys
import OpenEXR

def main():
    arguements = sys.argv[1:]
    if len(arguements) >= 2:
        input = arguements[0]
        output = arguements[1]
        with OpenEXR.File(input) as infile:
            RGB = infile.channels()["RGBA"].pixels
            height, width, _ = RGB.shape
            st = StructuredTensor(RGB)
            for y in range(height):
                for x in range(width):
                    st.process([x,y])
            ka = KuwaharaAnisotropic(RGB, st.getDst())
            ka.multiProcessAll()
            RGB = ka.getDst()
            infile.write(output)

    else:
        print("Wrong Arguements")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()