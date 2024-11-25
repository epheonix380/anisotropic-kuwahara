import numpy as np
from numba import cuda, jit


@jit
def mult_vec4(x: float, y: np.ndarray):
    return np.array([
        x*y[0],
        x*y[1],
        x*y[2],
        x*y[3],

    ])


@cuda.jit
def kuwa_cuda_1(src: np.ndarray, structured_tensor: np.ndarray, dst, size: float, eccentricity: float, sharpness: float):
    y, x = cuda.grid(2)
    if (y >= src.shape[0]) or (x >= src.shape[1]):
        return
    dxdx = structured_tensor[y][x][0]
    dxdy = structured_tensor[y][x][1]
    dydy = structured_tensor[y][x][2]

    eigenvalue_first_term = (dxdx + dydy) / 2.0
    eigenvalue_square_root_term = np.sqrt((dxdx - dydy)*(dxdx - dydy) + 4.0 * (dxdy*dxdy)) / 2.0
    first_eigenvalue = eigenvalue_first_term + eigenvalue_square_root_term
    second_eigenvalue = eigenvalue_first_term - eigenvalue_square_root_term

    eigenvector = np.array([first_eigenvalue - dxdx, -dxdy])
    eigenvector_length = np.linalg.norm(eigenvector)
    unit_eigenvector = eigenvector / eigenvector_length if eigenvector_length != 0.0 else np.array([1.0, 0.0])

    eigenvalue_sum = first_eigenvalue + second_eigenvalue
    eigenvalue_difference = first_eigenvalue - second_eigenvalue
    anisotropy = eigenvalue_difference / eigenvalue_sum if eigenvalue_sum > 0.0 else 0.0

    
    radius = max(0.0, size)
    if radius == 0.0:
        return src[y, x]
        

    eccentricity_clamp = min(eccentricity, 0.95)
    eccentric_adj = (1.0 - eccentricity_clamp) * 10.0
    ellipse_width_factor = (eccentric_adj + anisotropy) / eccentric_adj

    ellipse_width = ellipse_width_factor * radius
    ellipse_height = radius / ellipse_width_factor

    cosine = unit_eigenvector[0]
    sine = unit_eigenvector[1]

    inverse_ellipse_matrix_1 = np.array([cosine / ellipse_width, sine / ellipse_width])
    inverse_ellipse_matrix_2 = np.array([-sine / ellipse_height, cosine / ellipse_height])

    ellipse_major_axis = ellipse_width * unit_eigenvector
    ellipse_minor_axis = np.array([ellipse_height * unit_eigenvector[1] * -1.0,
                                ellipse_height * unit_eigenvector[0]])

    ellipse_bounds = [np.ceil(np.sqrt(ellipse_major_axis[0]*ellipse_major_axis[0] + ellipse_minor_axis[0]*ellipse_minor_axis[0])), np.sqrt(ellipse_major_axis[1]*ellipse_major_axis[1] + ellipse_minor_axis[1]*ellipse_minor_axis[1])]

    number_of_sectors = 8
    sector_center_overlap_parameter = 2.0 / radius
    sector_envelope_angle = ((3.0 / 2.0) * np.pi) / number_of_sectors
    cross_sector_overlap_parameter = (sector_center_overlap_parameter + 
                                    np.cos(sector_envelope_angle)) / (np.sin(sector_envelope_angle)*np.sin(sector_envelope_angle))

    weighted_mean_of_squared_color_of_sectors = [np.zeros(4) for _ in range(number_of_sectors)]
    weighted_mean_of_color_of_sectors = [np.zeros(4) for _ in range(number_of_sectors)]
    sum_of_weights_of_sectors = [0.0 for _ in range(number_of_sectors)]

    center_color = src[y][x]
    center_color_squared = np.dot(center_color, center_color)
    center_weight_b = 1.0 / number_of_sectors
    weighted_center_color = np.dot(np.array([center_weight_b,center_weight_b,center_weight_b,center_weight_b]), center_color)
    weighted_center_color_squared =np.dot(np.array([center_weight_b,center_weight_b,center_weight_b,center_weight_b]),center_color_squared)

    for i in range(number_of_sectors):
        weighted_mean_of_squared_color_of_sectors[i] = weighted_center_color_squared
        weighted_mean_of_color_of_sectors[i] = weighted_center_color
        sum_of_weights_of_sectors[i] = center_weight_b

def kuwaCuda(src: np.ndarray, structured_tensor: np.ndarray, size: float=10.0, sharpness: float=1.0, eccentricity: float=0.5) -> np.ndarray:
    height, width, depth = src.shape
    dst = cuda.device_array((height,width,depth), dtype=np.float32)
    print(src.shape)
    print(structured_tensor.shape)
    print(dst.shape)

    G = np.empty(dst.shape)

    #threadsperblock=32
    #blockspergrid = (G.size + (threadsperblock - 1))
    # Thread distribution parameters, has to do with GPU architecture
    threads_per_block = np.array([16,16])
    blocks_per_grid   = np.ceil(dst.shape[:2]/threads_per_block).astype(int)

    # Now start the kernel
    kuwa_cuda_1[tuple(blocks_per_grid), tuple(threads_per_block)](src, structured_tensor, dst, size, eccentricity, sharpness)
