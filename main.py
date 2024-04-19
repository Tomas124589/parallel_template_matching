import time

import cv2
import numpy as np
from PIL import Image
from numba import njit


def rgb2gray(matrix):
    gray = np.dot(matrix[..., :3], [0.2989, 0.5870, 0.1140])
    return gray[:, :, np.newaxis]


def template_match_opencv(image_src, template_src, show):
    image = cv2.imread(image_src, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_src, cv2.IMREAD_GRAYSCALE)

    templ_width, templ_height = template.shape[::-1]

    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if show:
        top_left = max_loc
        bottom_right = (top_left[0] + templ_width, top_left[1] + templ_height)

        cv2.rectangle(image, top_left, bottom_right, 255, 2)

        cv2.namedWindow('opencv', cv2.WINDOW_NORMAL)
        cv2.imshow('opencv', image)

        cv2.waitKey()

    return max_loc


@njit(parallel=True)
def normalized_cross_corellation(image: np.ndarray, template: np.ndarray):
    img_height, img_width, img_channels = image.shape
    templ_height, templ_width, templ_channels = template.shape

    shape = (img_height - templ_height + 1, img_width - templ_width + 1)
    template_mean = np.mean(template)
    template_diff = template - template_mean
    template_sum_sqrt = np.sum((template - template_mean) ** 2)

    max_result = 0.0
    max_x = max_y = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            window = image[x:x + templ_height, y:y + templ_width]
            window_mean = np.mean(window)
            window_diff = window - window_mean

            numerator = np.sum(window_diff * template_diff)
            denominator = np.sqrt(np.sum(window_diff ** 2) * template_sum_sqrt)

            result = numerator / denominator

            if result > max_result:
                max_x = x
                max_y = y
                max_result = result

    return max_y, max_x


if __name__ == '__main__':
    img_src = 'samples/02/source.png'
    template_src = 'samples/02/template.png'

    start = time.time()
    print(template_match_opencv(img_src, template_src, False))
    print("opencv took {} secs".format(time.time() - start))

    start = time.time()

    image = rgb2gray(np.array(Image.open(img_src)))
    template = rgb2gray(np.array(Image.open(template_src)))

    print(normalized_cross_corellation(image, template))
    print("nccor took {} secs".format(time.time() - start))
