from .AnisotropicKuwahara import KuwaharaAnisotropic
import sys


def main():
    arguements = sys.argv[1:]
    if len(arguements) >= 2:
        input = arguements[0]
        output = arguements[1]
        

    else:
        print("Wrong Arguements")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()