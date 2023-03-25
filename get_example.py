import cv2
import numpy as np
from glob import glob
import random


def get_test(path):
    rho = 32
    patch_size = 128
    height = 240
    width = 320
    color_image = cv2.imread(path)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (width, height))
    # cv2.imshow("gray", gray_image)
    # cv2.waitKey(0)
    # points
    y = random.randint(rho, height - rho - patch_size)  # row
    x = random.randint(rho, width - rho - patch_size)  # col
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    four_points_array = np.array(four_points)
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute H
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = np.linalg.inv(H)
    inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))

    # cv2.imshow("inv_warp", inv_warped_image)
    # cv2.waitKey(0)
    # re_warp = cv2.warpPerspective(inv_warped_image, H, (width, height))
    # cv2.imshow("re_warp", re_warp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # grab image patches
    original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
    # make into dataset
    training_image = np.dstack((original_patch, warped_patch))
    val_image = training_image.reshape((1, 128, 128, 2))

    return color_image, H_inverse, val_image, four_points_array


if __name__ == '__main__':
    get_test("i5.ppm")

