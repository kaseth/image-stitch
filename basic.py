import cv2
import numpy as np
import tensorflow.keras.backend as k
from Net_homography import euclidean_distance
from img_input import img_input
import tensorflow as tf


k.clear_session()
my_model = tf.keras.models.load_model("ho_model.h5", custom_objects={'euclidean_distance': euclidean_distance})

# Load the input images
img1 = cv2.imread('t1.jpg')
if img1.shape[0] > 1000:
    img1 = cv2.resize(img1, (int(960/img1.shape[0] * img1.shape[1]), 960))

cv2.imshow("img1", img1)
cv2.waitKey(0)
img2 = cv2.imread('t.jpg')
if img2.shape[0] > 1000:
    img2 = cv2.resize(img2, (int(960/img1.shape[0] * img1.shape[1]), 960))

# img2 = cv2.resize(img2, (1000, 1000))
# cv2.imshow("img2", img2)
# cv2.waitKey(0)

# Convert the input images to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

input_pair, four_points = img_input(img1_gray, img2_gray)
labels = my_model.predict(input_pair)
k.clear_session()
label = np.int32(labels.reshape((4, 2)))
perturbed_points = np.subtract(four_points, label)
H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_points))


# Create a SIFT object and detect keypoints and descriptors in both images
sift = cv2.xfeatures2d.SURF_create()
# sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Create a BFMatcher object and match the descriptors
bf = cv2.BFMatcher(crossCheck=False)

matches = bf.knnMatch(des1, des2, k=2)

# Sort the matches by distance
# matches = sorted(matches, key=lambda x: x.distance)

good_matches = []
ratio = 0.5

for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append(m)

# Compute the homography matrix from the top matche
if len(good_matches) >= 4:
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
else:
    print("no enough matches！")
    exit()


print("H", H )
print("M", M)
# Warp the second image onto the first using the homography matrix
warp = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
result = warp.copy()
# cv2.imshow("warp", result)
# show_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
# cv2.imshow("matches", show_match)
# cv2.waitKey(0)
result[0:img1.shape[0], 0:img1.shape[1]-20] = img1[:, 0:img1.shape[1]-20]
cv2.imshow("warp", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = np.zeros(result.shape[:2], dtype=np.uint8)
mask[:, 0:img1.shape[1]] = 255
cv2.imshow("mask", mask)
cv2.waitKey(0)
result = cv2.seamlessClone(img1, result, mask, (img1.shape[1]//2, img1.shape[0]//2), cv2.NORMAL_CLONE)
cv2.imshow("blend", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

stitched_img = result
# Find the bounding box of the stitched image
# height, width, channels = stitched_img.shape
# min_x, min_y = width, height
# max_x = max_y = 0
# for y in range(height):
#     for x in range(width):
#         if not all(stitched_img[y, x] == [0, 0, 0]):
#             min_x = min(x, min_x)
#             min_y = min(y, min_y)
#             max_x = max(x, max_x)
#             max_y = max(y, max_y)
#
# bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
#
# # Crop the stitched image to the bounding box
# stitched_img_cropped = stitched_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
# cv2.imshow("crop", stitched_img_cropped)
# cv2.waitKey(0)

# Find the contours of the non-black regions
gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding rectangle of the non-black regions
# cnt = max(contours, key=cv2.contourArea)
contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
# cont = cv2.drawContours(img1.copy(), contours, 0, (0, 0, 255), 2)
# cv2.imshow("cont", cont)
# cv2.waitKey(0)

rect = x_cnt, y_cbt, w, h = cv2.boundingRect(contours[0])
rect_img = stitched_img.copy()
cv2.rectangle(rect_img, (0, 0), (w, h), (255, 0, 0), 1)
cv2.imshow("rect", rect_img)
cv2.waitKey(0)


# Crop the stitched image to the bounding rectangle
stitched_img_cropped_rect = stitched_img[:h-40, :w-40]


cv2.imshow("pano", stitched_img_cropped_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('result.jpg', result)
