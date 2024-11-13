import numpy as np

def structure_tensor(src):
    """
    Computes the structure tensor of the image using a Dirac delta window function as described in
    section "3.2 Local Structure Estimation" of the paper: Kyprianidis, Jan Eric. 
    "Image and video abstraction by multi-scale anisotropic Kuwahara filtering." 2011.
    
    The structure tensor should then be smoothed using a Gaussian function to eliminate high frequency details.
    
    Adapted from the Blender GLSL script (Copyright: 2023 Blender Authors)
    https://projects.blender.org/blender/blender/src/source/blender/compositor/realtime_compositor/shaders/
    compositor_kuwahara_anisotropic_compute_structure_tensor.glsl
    
    Python implementation by AI Assistant, 2024
    """
    
    # The weight kernels of the filter optimized for rotational symmetry 
    # described in section "3.2.1 Gradient Calculation".
    corner_weight = 0.182
    center_weight = 1.0 - 2.0 * corner_weight
    
    # Pad the input image to handle edge cases
    padded_src = np.pad(src, ((1, 1), (1, 1), (0, 0)), mode='edge')
    
    # Compute partial derivatives
    x_partial_derivative = (
        -corner_weight * padded_src[1:-1, :-2] +
        -center_weight * padded_src[1:-1, 1:-1] +
        -corner_weight * padded_src[1:-1, 2:] +
        corner_weight * padded_src[1:-1, 2:] +
        center_weight * padded_src[1:-1, 1:-1] +
        corner_weight * padded_src[1:-1, :-2]
    )
    
    y_partial_derivative = (
        corner_weight * padded_src[:-2, 1:-1] +
        center_weight * padded_src[1:-1, 1:-1] +
        corner_weight * padded_src[2:, 1:-1] +
        -corner_weight * padded_src[2:, 1:-1] +
        -center_weight * padded_src[1:-1, 1:-1] +
        -corner_weight * padded_src[:-2, 1:-1]
    )
    
    # Compute structure tensor components
    dxdx = np.sum(x_partial_derivative * x_partial_derivative, axis=-1)
    dxdy = np.sum(x_partial_derivative * y_partial_derivative, axis=-1)
    dydy = np.sum(y_partial_derivative * y_partial_derivative, axis=-1)
    
    # Stack the components into a 3-channel image
    dst = np.stack([dxdx, dxdy, dydy], axis=-1)
    
    return dst

