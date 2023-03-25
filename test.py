import numpy as np
import cv2
import random
import tensorflow as tf
import get_example
import tensorflow.keras.backend as k

from Net_homography import homography_net


def patch_generate():
    img1 = cv2.imread('i1.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (320, 240))
    cv2.imshow("img1", img1)
    cv2.waitKey(0)
    rho = 32
    patch_size = 128
    top_point = (64, 64)
    left_point = (patch_size + 64, 64)
    bottom_point = (patch_size + 64, patch_size + 64)
    right_point = (64, patch_size + 64)
    annotated_image = img1.copy()
    four_points = [top_point, left_point, bottom_point, right_point]
    cv2.polylines(annotated_image, np.int32([four_points]), 1, (255, 0, 0))
    cv2.imshow("annotated", annotated_image)
    cv2.waitKey(0)
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
    perturbed_annotated_image = img1.copy()
    cv2.polylines(perturbed_annotated_image, np.int32([perturbed_four_points]), 1, (255, 0, 0))
    cv2.imshow("annotated", perturbed_annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = np.linalg.inv(H)
    warped_image = cv2.warpPerspective(img1, H_inverse, (320, 240))
    Ip1 = img1[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    cv2.imshow("Ip1", Ip1)
    cv2.waitKey(0)
    cv2.imshow("ip2", Ip2)
    cv2.waitKey(0)
    training_image = np.dstack((Ip1, Ip2))
    print(training_image.shape)
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    print(H_four_points)


def model_test():
    tf.keras.backend.clear_session()
    # my_model = tf.keras.models.load_model("ho_model.h5", custom_objects={'euclidean_distance': euclidean_distance})
    my_model = homography_net()
    my_model.load_weights("./checkpoint/weights.50-42.33.h5")
    # my_model.summary()
    img, h_inv, val_img, four_points = get_example.get_test("i5.ppm")
    four_points = four_points.reshape((1, 4, 2))
    print("read finish")
    line_img = cv2.polylines(img, four_points, 1, (0, 0, 255), 2)
    warp_img = cv2.warpPerspective(line_img, h_inv, (img.shape[1], img.shape[0]))
    cv2.imshow("warp", warp_img)
    cv2.waitKey(0)
    labels = my_model.predict(val_img)
    label = np.int32(labels.reshape((4, 2)))
    perturbed_points = np.subtract(four_points, label)
    four_points = four_points.reshape((4, 2))
    H = cv2.getPerspectiveTransform(np.float32(four_points),  np.float32(perturbed_points))
    H_inv = np.linalg.inv(H)
    re_warp = cv2.warpPerspective(warp_img, H_inv, (warp_img.shape[1], warp_img.shape[0]))
    cv2.imshow("re_warp", re_warp)
    cv2.waitKey(0)
    perturbed_points = perturbed_points.reshape((1, 4, 2))
    warp_img = cv2.polylines(warp_img, perturbed_points, 1, (255, 0, 0), 2)

    cv2.imshow("origin", img)
    cv2.waitKey(0)
    cv2.imshow("perturb", warp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_test()



