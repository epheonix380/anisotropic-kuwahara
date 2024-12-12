import processing as ip
import numpy as np

def renderFromNDarray(image: np.ndarray, radius:int = 5):
    print(f"radius is {radius}")
    fragment_code = ip.readFile('shader_code/generalized_kuwahara.frag')
    shader = ip.Shader(fragment=fragment_code)
    shader.setUniforms(radius=radius)
    flipped_image = np.flip(image, 0)
    processed_texture = shader.run(flipped_image)
    return processed_texture.toArray()
