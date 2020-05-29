import os
import random

import cv2
import numpy as np

root_dir = "../data/kaist/images/"


def calc_stats():
    rgb_images = []
    ir_images = []
    mean = []
    std = []
    set_list = os.listdir(root_dir)
    for s_path in [os.path.join(root_dir, s) for s in set_list]:
        vid_list = os.listdir(s_path)
        for v_path in [os.path.join(s_path, b) for b in vid_list]:
            img_list = os.listdir(os.path.join(v_path, "visible"))
            for img_name in img_list:
                if random.uniform(0, 1) > 0.95:
                    rgb_img = cv2.imread(os.path.join(v_path, "visible", img_name))
                    ir_img = cv2.imread(os.path.join(v_path, "lwir", img_name), cv2.IMREAD_GRAYSCALE)
                    rgb_images.append(rgb_img)
                    ir_images.append(ir_img)
    print(len(rgb_images))

    # calc mean
    total_red = 0
    total_green = 0
    total_blue = 0
    total_intensity = 0
    for index in range(len(rgb_images)):
        img = rgb_images[index]
        ir_img = ir_images[index]

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                ir_pixel = ir_img[i, j]

                red = pixel[0]
                total_red += red

                green = pixel[1]
                total_green += green

                blue = pixel[2]
                total_blue += blue

                intensity = ir_pixel
                total_intensity += intensity
    denominator = len(rgb_images) * 511 * 511 * 255

    mean.append(total_red / denominator)
    mean.append(total_green / denominator)
    mean.append(total_blue / denominator)
    mean.append(total_intensity / denominator)
    print("Mean: ", mean)

    std_sum_red = 0
    std_sum_green = 0
    std_sum_blue = 0
    std_sum_ir = 0
    for index in range(len(rgb_images)):
        img = rgb_images[index]
        ir_img = ir_images[index]

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                ir_pixel = ir_img[i, j]

                red = pixel[0] / 255
                std_sum_red += (mean[0] - red) ** 2

                green = pixel[1] / 255
                std_sum_green += (mean[1] - green) ** 2

                blue = pixel[2] / 255
                std_sum_blue += (mean[2] - blue) ** 2

                intensity = ir_pixel / 255
                std_sum_ir += (mean[3] - intensity) ** 2
    std_denom = len(rgb_images) * 511 * 511
    std.append(np.sqrt(std_sum_red / std_denom))
    std.append(np.sqrt(std_sum_green / std_denom))
    std.append(np.sqrt(std_sum_blue / std_denom))
    std.append(np.sqrt(std_sum_ir / std_denom))
    print("Std: ", std)

    all_lst = []
    for i in range(len(rgb_images)):
        res = np.reshape(rgb_images[i], (3, 511, 511))
        added = np.append(res, ir_images[i])
        added = np.reshape(added, (4, 511, 511))
        all_lst.append(added)

    # Must has (N x 4 x 511 x 511) ndarray
    print(added)


if __name__ == "__main__":
    calc_stats()
