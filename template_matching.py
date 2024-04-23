import numpy as np
from numba import njit


@njit("Tuple((int64,int64))(float64[:,:,:], float64[:,:,:], int64)", parallel=True, cache=True)
def normalized_cross_correlation(image: np.ndarray, template: np.ndarray, step: int = 1):
    img_height, img_width, img_channels = image.shape
    templ_height, templ_width, templ_channels = template.shape

    shape = (img_height - templ_height) // step + 1, (img_width - templ_width) // step + 1
    template_sum_sqrt = np.sum(template ** 2)

    max_val = 0.0
    pos = 0, 0
    for x in range(shape[0]):
        x_step = x * step
        x_step_templ_height = x_step + templ_height

        for y in range(shape[1]):
            window = image[x_step:x_step_templ_height, y * step:y * step + templ_width]

            val = np.sum(window * template) / np.sqrt(np.sum(window ** 2) * template_sum_sqrt)
            if val > max_val:
                pos = y * step, x_step
                max_val = val

    return pos


def rgb2gray(matrix):
    gray = np.dot(matrix[..., :3], [0.2989, 0.5870, 0.1140])
    return gray[:, :, np.newaxis]
