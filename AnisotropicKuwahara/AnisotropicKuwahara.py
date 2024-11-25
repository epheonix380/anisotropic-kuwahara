import numpy as np
from numba import jit, cuda
from multiprocessing import Pool
from .Filter import Filter
class KuwaharaAnisotropic(Filter):
    def __init__(self, structure_tensor: np.ndarray, src: np.ndarray=None, size: float=10.0, sharpness: float=1.0, eccentricity: float=0.5):
        self.size = size
        self.sharpness = sharpness
        self.eccentricity = eccentricity
        self.structure_tensor = structure_tensor
        if src is not None:
            height, width, _ = src.shape
            self.height = height
            self.width = width
            self.dst = np.zeros(shape=(height, width, 4))
            self.src = src

    @staticmethod
    def square(x: float) -> float:
        return x * x
    
    
    
    def processWithCuda(self) -> np.ndarray:
        @cuda.jit
        def interal_cuda_helper(src, height, width, structured_tensor, dst):
            y = 0
            for y in range(height):
                for x in range(width):
                    pos = [y, x]
                    structure_tensor = structured_tensor
                    src = src
                    encoded_structure_tensor = structure_tensor[pos[0], pos[1]]

                    dxdx = encoded_structure_tensor[0]
                    dxdy = encoded_structure_tensor[1]
                    dydy = encoded_structure_tensor[2]

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

                    radius = max(0.0, self.size)
                    if radius == 0.0:
                        return src[pos[0], pos[1]]
                        

                    eccentricity_clamp = min(self.eccentricity, 0.95)
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

                    center_color = src[pos[0], pos[1]]
                    center_color_squared = center_color * center_color
                    center_weight_b = 1.0 / number_of_sectors
                    weighted_center_color = center_color * center_weight_b
                    weighted_center_color_squared = center_color_squared * center_weight_b

                    for i in range(number_of_sectors):
                        weighted_mean_of_squared_color_of_sectors[i] = weighted_center_color_squared
                        weighted_mean_of_color_of_sectors[i] = weighted_center_color
                        sum_of_weights_of_sectors[i] = center_weight_b

                    for j in range(int(ellipse_bounds[1]) + 1):
                        for i in range(-int(ellipse_bounds[0]), int(ellipse_bounds[0]) + 1):
                            if pos[0] + i >= self.height:
                                continue
                            if pos[0] - i < 0:
                                continue
                            # This is necessary due to potentially negative i
                            if pos[0] - i >= self.height:
                                continue
                            # This is necessary due to potentially negative i
                            if pos[0] + i < 0:
                                continue
                            if pos[1]+j >= self.width:
                                continue
                            if pos[1]-j < 0:
                                continue
                            if j == 0 and i <= 0:
                                continue

                            disk_point = np.array([inverse_ellipse_matrix_1[0] * i + inverse_ellipse_matrix_1[1] * j,
                                                inverse_ellipse_matrix_2[0] * i + inverse_ellipse_matrix_2[1] * j])

                            disk_point_length_squared = np.dot(disk_point, disk_point)
                            if disk_point_length_squared > 1.0:
                                continue

                            sector_weights = [0.0] * 8

                            polynomial = sector_center_overlap_parameter - cross_sector_overlap_parameter * disk_point * disk_point
                            sector_weights[0] = (max(0.0, disk_point[1] + polynomial[0]))*(max(0.0, disk_point[1] + polynomial[0]))
                            sector_weights[2] = (max(0.0, -disk_point[0] + polynomial[1]))*(max(0.0, -disk_point[0] + polynomial[1]))
                            sector_weights[4] = (max(0.0, -disk_point[1] + polynomial[0]))*(max(0.0, -disk_point[1] + polynomial[0]))
                            sector_weights[6] = (max(0.0, disk_point[0] + polynomial[1]))*(max(0.0, disk_point[0] + polynomial[1]))

                            M_SQRT1_2 = 1.0 / np.sqrt(2.0)
                            rotated_disk_point = M_SQRT1_2 * np.array([disk_point[0] - disk_point[1], disk_point[0] + disk_point[1]])

                            rotated_polynomial = sector_center_overlap_parameter - cross_sector_overlap_parameter * rotated_disk_point * rotated_disk_point
                            sector_weights[1] = (max(0.0, rotated_disk_point[1] + rotated_polynomial[0]))*(max(0.0, rotated_disk_point[1] + rotated_polynomial[0]))
                            sector_weights[3] = (max(0.0, -rotated_disk_point[0] + rotated_polynomial[1]))*(max(0.0, -rotated_disk_point[0] + rotated_polynomial[1]))
                            sector_weights[5] = (max(0.0, -rotated_disk_point[1] + rotated_polynomial[0]))*(max(0.0, -rotated_disk_point[1] + rotated_polynomial[0]))
                            sector_weights[7] = (max(0.0, rotated_disk_point[0] + rotated_polynomial[1]))*(max(0.0, rotated_disk_point[0] + rotated_polynomial[1]))

                            sector_weights_sum = sum(sector_weights)
                            radial_gaussian_weight = np.exp(-np.pi * disk_point_length_squared) / sector_weights_sum

                            upper_color = src[pos[0] + i, pos[1] + j]
                            lower_color = src[pos[0] - i, pos[1] - j]
                            upper_color_squared = upper_color * upper_color
                            lower_color_squared = lower_color * lower_color

                            for k in range(number_of_sectors):
                                weight = sector_weights[k] * radial_gaussian_weight

                                upper_index = k
                                sum_of_weights_of_sectors[upper_index] += weight
                                weighted_mean_of_color_of_sectors[upper_index] += upper_color * weight
                                weighted_mean_of_squared_color_of_sectors[upper_index] += upper_color_squared * weight

                                lower_index = (k + number_of_sectors // 2) % number_of_sectors
                                sum_of_weights_of_sectors[lower_index] += weight
                                weighted_mean_of_color_of_sectors[lower_index] += lower_color * weight
                                weighted_mean_of_squared_color_of_sectors[lower_index] += lower_color_squared * weight

                    sum_of_weights = 0.0
                    weighted_sum = np.zeros(4)
                    for i in range(number_of_sectors):
                        weighted_mean_of_color_of_sectors[i] /= sum_of_weights_of_sectors[i]
                        weighted_mean_of_squared_color_of_sectors[i] /= sum_of_weights_of_sectors[i]

                        color_mean = weighted_mean_of_color_of_sectors[i]
                        squared_color_mean = weighted_mean_of_squared_color_of_sectors[i]
                        color_variance = np.abs(squared_color_mean - color_mean * color_mean)

                        standard_deviation = np.dot(np.sqrt(color_variance[:3]), np.array([1.0, 1.0, 1.0]))
                        
                        normalized_sharpness = self.sharpness * 10.0
                        weight = 1.0 / np.power(max(0.02, standard_deviation), normalized_sharpness)

                        sum_of_weights += weight
                        weighted_sum += color_mean * weight

                    weighted_sum /= sum_of_weights
                    dst[pos[0]][pos[1]] =  np.array([weighted_sum[0], weighted_sum[1], weighted_sum[2], center_color[3]])
        threadsperblock = 32 

        # Calculate the number of thread blocks in the grid
        blockspergrid = (self.src.size + (threadsperblock - 1)) // threadsperblock

        # Now start the kernel
        interal_cuda_helper[blockspergrid, threadsperblock](self.src,self.height, self.width, self.structure_tensor, self.dst)


    def process(self, pos: list[int]) -> np.ndarray:
        structure_tensor = self.structure_tensor
        src = self.src
        encoded_structure_tensor = structure_tensor[pos[0], pos[1]]

        dxdx = encoded_structure_tensor[0]
        dxdy = encoded_structure_tensor[1]
        dydy = encoded_structure_tensor[2]

        eigenvalue_first_term = (dxdx + dydy) / 2.0
        eigenvalue_square_root_term = np.sqrt(self.square(dxdx - dydy) + 4.0 * self.square(dxdy)) / 2.0
        first_eigenvalue = eigenvalue_first_term + eigenvalue_square_root_term
        second_eigenvalue = eigenvalue_first_term - eigenvalue_square_root_term

        eigenvector = np.array([first_eigenvalue - dxdx, -dxdy])
        eigenvector_length = np.linalg.norm(eigenvector)
        unit_eigenvector = eigenvector / eigenvector_length if eigenvector_length != 0.0 else np.array([1.0, 0.0])

        eigenvalue_sum = first_eigenvalue + second_eigenvalue
        eigenvalue_difference = first_eigenvalue - second_eigenvalue
        anisotropy = eigenvalue_difference / eigenvalue_sum if eigenvalue_sum > 0.0 else 0.0

        radius = max(0.0, self.size)
        if radius == 0.0:
            return src[pos[0], pos[1]]
            

        eccentricity_clamp = min(self.eccentricity, 0.95)
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

        ellipse_bounds = [np.ceil(np.sqrt(self.square(ellipse_major_axis[0]) + self.square(ellipse_minor_axis[0]))), np.sqrt(self.square(ellipse_major_axis[1]) + self.square(ellipse_minor_axis[1]))]

        number_of_sectors = 8
        sector_center_overlap_parameter = 2.0 / radius
        sector_envelope_angle = ((3.0 / 2.0) * np.pi) / number_of_sectors
        cross_sector_overlap_parameter = (sector_center_overlap_parameter + 
                                        np.cos(sector_envelope_angle)) / self.square(np.sin(sector_envelope_angle))

        weighted_mean_of_squared_color_of_sectors = [np.zeros(4) for _ in range(number_of_sectors)]
        weighted_mean_of_color_of_sectors = [np.zeros(4) for _ in range(number_of_sectors)]
        sum_of_weights_of_sectors = [0.0 for _ in range(number_of_sectors)]

        center_color = src[pos[0], pos[1]]
        center_color_squared = center_color * center_color
        center_weight_b = 1.0 / number_of_sectors
        weighted_center_color = center_color * center_weight_b
        weighted_center_color_squared = center_color_squared * center_weight_b

        for i in range(number_of_sectors):
            weighted_mean_of_squared_color_of_sectors[i] = weighted_center_color_squared
            weighted_mean_of_color_of_sectors[i] = weighted_center_color
            sum_of_weights_of_sectors[i] = center_weight_b

        for j in range(int(ellipse_bounds[1]) + 1):
            for i in range(-int(ellipse_bounds[0]), int(ellipse_bounds[0]) + 1):
                if pos[0] + i >= self.height:
                    continue
                if pos[0] - i < 0:
                    continue
                # This is necessary due to potentially negative i
                if pos[0] - i >= self.height:
                    continue
                # This is necessary due to potentially negative i
                if pos[0] + i < 0:
                    continue
                if pos[1]+j >= self.width:
                    continue
                if pos[1]-j < 0:
                    continue
                if j == 0 and i <= 0:
                    continue

                disk_point = np.array([inverse_ellipse_matrix_1[0] * i + inverse_ellipse_matrix_1[1] * j,
                                    inverse_ellipse_matrix_2[0] * i + inverse_ellipse_matrix_2[1] * j])

                disk_point_length_squared = np.dot(disk_point, disk_point)
                if disk_point_length_squared > 1.0:
                    continue

                sector_weights = [0.0] * 8

                polynomial = sector_center_overlap_parameter - cross_sector_overlap_parameter * disk_point * disk_point
                sector_weights[0] = self.square(max(0.0, disk_point[1] + polynomial[0]))
                sector_weights[2] = self.square(max(0.0, -disk_point[0] + polynomial[1]))
                sector_weights[4] = self.square(max(0.0, -disk_point[1] + polynomial[0]))
                sector_weights[6] = self.square(max(0.0, disk_point[0] + polynomial[1]))

                M_SQRT1_2 = 1.0 / np.sqrt(2.0)
                rotated_disk_point = M_SQRT1_2 * np.array([disk_point[0] - disk_point[1], disk_point[0] + disk_point[1]])

                rotated_polynomial = sector_center_overlap_parameter - cross_sector_overlap_parameter * rotated_disk_point * rotated_disk_point
                sector_weights[1] = self.square(max(0.0, rotated_disk_point[1] + rotated_polynomial[0]))
                sector_weights[3] = self.square(max(0.0, -rotated_disk_point[0] + rotated_polynomial[1]))
                sector_weights[5] = self.square(max(0.0, -rotated_disk_point[1] + rotated_polynomial[0]))
                sector_weights[7] = self.square(max(0.0, rotated_disk_point[0] + rotated_polynomial[1]))

                sector_weights_sum = sum(sector_weights)
                radial_gaussian_weight = np.exp(-np.pi * disk_point_length_squared) / sector_weights_sum

                upper_color = src[pos[0] + i, pos[1] + j]
                lower_color = src[pos[0] - i, pos[1] - j]
                upper_color_squared = upper_color * upper_color
                lower_color_squared = lower_color * lower_color

                for k in range(number_of_sectors):
                    weight = sector_weights[k] * radial_gaussian_weight

                    upper_index = k
                    sum_of_weights_of_sectors[upper_index] += weight
                    weighted_mean_of_color_of_sectors[upper_index] += upper_color * weight
                    weighted_mean_of_squared_color_of_sectors[upper_index] += upper_color_squared * weight

                    lower_index = (k + number_of_sectors // 2) % number_of_sectors
                    sum_of_weights_of_sectors[lower_index] += weight
                    weighted_mean_of_color_of_sectors[lower_index] += lower_color * weight
                    weighted_mean_of_squared_color_of_sectors[lower_index] += lower_color_squared * weight

        sum_of_weights = 0.0
        weighted_sum = np.zeros(4)
        for i in range(number_of_sectors):
            weighted_mean_of_color_of_sectors[i] /= sum_of_weights_of_sectors[i]
            weighted_mean_of_squared_color_of_sectors[i] /= sum_of_weights_of_sectors[i]

            color_mean = weighted_mean_of_color_of_sectors[i]
            squared_color_mean = weighted_mean_of_squared_color_of_sectors[i]
            color_variance = np.abs(squared_color_mean - color_mean * color_mean)

            standard_deviation = np.dot(np.sqrt(color_variance[:3]), np.array([1.0, 1.0, 1.0]))

            normalized_sharpness = self.sharpness * 10.0
            weight = 1.0 / np.power(max(0.02, standard_deviation), normalized_sharpness)

            sum_of_weights += weight
            weighted_sum += color_mean * weight

        weighted_sum /= sum_of_weights
        return  np.array([weighted_sum[0], weighted_sum[1], weighted_sum[2], center_color[3]])

