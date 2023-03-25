import cv2
import numpy as np


def img_input(img1, img2):
    img1 = cv2.resize(img1, (320, 240))
    img2 = cv2.resize(img2, (320, 240))
    patch_size = 128
    top_point = (64, 64)
    left_point = (patch_size + 64, 64)
    bottom_point = (patch_size + 64, patch_size + 64)
    right_point = (64, patch_size + 64)
    four_points = [top_point, left_point, bottom_point, right_point]
    Ip1 = img1[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = img2[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    # cv2.imshow("Ip1", Ip1)
    # cv2.waitKey(0)
    # cv2.imshow("ip2", Ip2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    input_pair = np.dstack((Ip1, Ip2))
    return input_pair, four_points


if __name__ == "__main__":
    img1 = cv2.imread('t1.jpg')
    img2 = cv2.imread('t2.jpg')
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    input_pair, four_points = img_input(img1_gray, img2_gray)
