import numpy as np
from numba import njit, prange


@njit("Tuple((int32,int32))(float64[:,:,:], float64[:,:,:], int64)", parallel=True, cache=True)
def normalized_cross_correlation(image: np.ndarray, template: np.ndarray, step: int = 1):
    img_height, img_width, img_channels = image.shape
    templ_height, templ_width, templ_channels = template.shape

    shape = (img_height - templ_height) // step + 1, (img_width - templ_width) // step + 1
    template_sum_sqrt = np.sum(template ** 2)

    max_result = np.zeros((image.shape[0],), dtype=np.float64)
    pos_x = np.zeros((image.shape[0],), dtype=np.int64)
    pos_y = np.zeros((image.shape[0],), dtype=np.int64)

    for x in prange(shape[0]):
        local_max = 0.0
        local_pos = (0, 0)

        x_step = x * step

        for y in prange(shape[1]):
            y_step = y * step
            window = image[x_step:x_step + templ_height, y_step:y_step + templ_width]

            val = np.sum(template * window) / np.sqrt(template_sum_sqrt * np.sum(window ** 2))
            if val > local_max:
                local_pos = x_step, y_step
                local_max = val

        max_result[x] = local_max
        pos_x[x], pos_y[x] = local_pos

    final_idx = np.argmax(max_result)
    return pos_y[final_idx], pos_x[final_idx]


def rgb2gray(matrix):
    gray = np.dot(matrix[..., :3], [0.2989, 0.5870, 0.1140])
    return gray[:, :, np.newaxis]
