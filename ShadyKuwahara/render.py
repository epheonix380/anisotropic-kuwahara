import processing as ip
import numpy as np

def renderFromNDarray(image: np.ndarray):
    # fragment_code = ip.readFile('shader_code/test_shader.glsl')
    # shader = ip.Shader(fragment=fragment_code)
    shader = ip.Shader(ip.GRAYSCALE)
    processed_texture = shader.run(image)
    processed_texture.save("output/test_output.png")
    return processed_texture.toArray()
