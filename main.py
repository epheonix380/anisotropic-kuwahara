
from AnisotropicKuwahara.utils.ImageUtil import plot_image, downsample_image, read_image, gaussian
from AnisotropicKuwahara.Tensor import StructuredTensor
from AnisotropicKuwahara.utils.TimeReport import TimeReport
from AnisotropicKuwahara.AnisotropicKuwahara import KuwaharaAnisotropic
from AnisotropicKuwahara.KuwaCuda import kuwaCuda

img = read_image("examples/shapes_all_colors.jpg")
downsampled_RGBA = img
# plot_image(downsampled_RGBA, title="Original Image, downsampled") # uncomment to see the image after downsampling
print("Starting Tensor")
st = StructuredTensor(downsampled_RGBA)
height, width, _ = downsampled_RGBA.shape
for y in range(height):
    for x in range(width):
        st.process([x,y])
print("Starting multi")
kuwaCuda(img, structured_tensor=st.getDst())
print("Multi finished")
