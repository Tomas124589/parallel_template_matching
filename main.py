import time

import cv2
import numpy as np
from PIL import Image
from numba import njit, set_num_threads


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


if __name__ == '__main__':
    img_src = 'samples/03/source.png'
    template_src = 'samples/03/template.png'

    image = rgb2gray(np.array(Image.open(img_src)))
    template = rgb2gray(np.array(Image.open(template_src)))

    thread_results = {}

    threads = [1, 2, 4, 6, 8]
    measurement_cnt = 5

    for t in threads:
        set_num_threads(t)
        ocv_time_total = 0
        nccor_time_total = 0

        for x in range(measurement_cnt):
            start = time.time()
            ocv_pos = template_match_opencv(img_src, template_src, False)
            opencv_end = time.time() - start
            ocv_time_total = ocv_time_total + opencv_end

            print("OpenCV {} took {:.4f} secs".format(ocv_pos, opencv_end))

            start = time.time()
            nccor_pos = normalized_cross_correlation(image, template, 16)
            nccor_end = time.time() - start
            nccor_time_total = nccor_time_total + nccor_end

            print("NCCOR {} took {:.4f} secs".format(nccor_pos, nccor_end))
            print("--------------------------------------------------")

        ocv_avg = ocv_time_total / measurement_cnt
        nccor_avg = nccor_time_total / measurement_cnt
        print("{} measures on {} threads".format(measurement_cnt, t))
        print("\t\t\tTotal(s)\tAVG(s)\tNCCOR/OpenCV\tOpenCV/NCCOR".format(measurement_cnt))
        print("OpenCV\t\t\t{:.4f}\t\t{:.4f}\t-\t\t\t\t{:.4f}"
              .format(ocv_time_total, ocv_avg, ocv_time_total / nccor_time_total))
        print("NCCOR\t\t\t{:.4f}\t\t{:.4f}\t{:.4f}\t\t\t-"
              .format(nccor_time_total, nccor_avg, nccor_time_total / ocv_time_total))

        thread_results[t] = {'total': nccor_time_total, 'avg': nccor_avg}

    print("================================================")
    print("Threads\tTotal(s)\tAVG(s)")
    for t_count, result in thread_results.items():
        print("{}\t\t{:.4f}\t\t{:.4f}".format(t_count, result['total'], result['avg']))
