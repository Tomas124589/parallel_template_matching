import numpy as np
from numba import njit


@njit("Tuple((int64,int64))(float64[:,:,:], float64[:,:,:], int64)", parallel=True, cache=True)
def normalized_cross_correlation(image: np.ndarray, template: np.ndarray, step: int = 1):
    img_height, img_width, img_channels = image.shape
    templ_height, templ_width, templ_channels = template.shape

    shape = ((img_height - templ_height) // step + 1, (img_width - templ_width) // step + 1)
    template_mean = np.mean(template)
    template_diff = template - template_mean
    template_sum_sqrt = np.sum((template - template_mean) ** 2)

    max_result = 0.0
    max_x = max_y = 0
    for x in range(shape[0]):
        x_step = x * step
        x_step_templ_height = x_step + templ_height

        for y in range(shape[1]):
            window = image[x_step:x_step_templ_height, y * step:y * step + templ_width]
            window_mean = np.mean(window)
            window_diff = window - window_mean

            numerator = np.sum(window_diff * template_diff)
            denominator = np.sqrt(np.sum(window_diff ** 2) * template_sum_sqrt)

            result = numerator / denominator

            if result > max_result:
                max_x = x_step
                max_y = y * step
                max_result = result

    return max_y, max_x


def rgb2gray(matrix):
    gray = np.dot(matrix[..., :3], [0.2989, 0.5870, 0.1140])
    return gray[:, :, np.newaxis]
