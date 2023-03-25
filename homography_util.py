import cv2
import numpy as np


# SIFT获取关键点和描述信息
def get_kp_des(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def get_match(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches found.")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def get_homography(src_pts, dst_pts):
    # Extract feature point pairs and their corresponding homography matrices
    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return h


if __name__ == '__main__':
    img1 = cv2.imread("image1.png")
    img2 = cv2.imread("image2.png")
    src, dst, h = get_homography(img1, img2)
    src = src[:, 0, :]
    print(src)
    print(dst)
    print(h)
