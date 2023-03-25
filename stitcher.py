import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from generater_dataset import *
from homography_util import *

path = "D:\\hpatches-sequences-release\\v"


# img1 = cv2.imread('image1.png')
# img2 = cv2.imread('image2.png')
# kp1, desc1 = get_kp_des(img1)
# kp2, desc2 = get_kp_des(img2)
# src_pts, dst_pts = get_match(kp1, desc1, kp2, desc2)
# H = get_homography(src_pts, dst_pts)
# src_pts = np.array(src_pts[:, 0, :])
# dst_pts = np.array(dst_pts[:, 0, :])
# print("*******")
# Define the model architecture
def homography_model(input_shape):
    input1 = tf.keras.Input(input_shape)
    input2 = tf.keras.Input(input_shape)

    x1 = tf.keras.layers.Flatten()(input1)
    x2 = tf.keras.layers.Flatten()(input2)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, )(x)

    output = tf.keras.layers.Reshape((8,))(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model


# Define the loss function
def homography_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 4, 2])
    y_pred = tf.reshape(y_pred, [-1, 4, 2])

    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss


# Prepare the data
# Load the dataset of pairs of images with known homographies


# Use a feature detector, such as SIFT or SURF, to extract feature points from each image
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, desc1 = sift.detectAndCompute(img1, None)
# kp2, desc2 = sift.detectAndCompute(img2, None)

# Find corresponding keypoints between the two images
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(desc1, desc2, k=2)
#
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)
#
# if len(good_matches) < 4:
#     print("Not enough good matches found.")
#     exit()

# Extract feature point pairs and their corresponding homography matrices
# src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


# Split the dataset into training, validation, and testing sets
# x1 = src_pts[:, 0, :]
# x2 = dst_pts[:, 0, :]
# y = H.flatten()
# match_list_src, match_list_dst, h_list = get_dataset(path)
data = np.load('data.npz')
match_list_src = data['match_list_src']
match_list_dst = data['match_list_dst']
h_list = data['h_list']
train_x1, test_x1, train_x2, test_x2, train_y, test_y = train_test_split(match_list_src, match_list_dst, h_list,
                                                                         test_size=0.2, random_state=10)
# train_x2, test_x2, train_y, test_y = train_test_split(match_list_dst, h_list, test_size=0.2, random_state=42)


# Define the model and compile it
input_shape = (50, 2)
model = homography_model(input_shape)
model.compile(optimizer='adam', loss=homography_loss)

# Train the model
batch_size = 16
epochs = 20
model.fit([train_x1, train_x2], train_y, validation_data=([test_x1, test_x2], test_y), batch_size=batch_size,
          epochs=epochs)

# Predict the homography matrices for the testing set
test_loss = model.evaluate([test_x1, test_x2], test_y, batch_size=batch_size)
print(f"Test loss: {test_loss:.4f}")

# model.save('my_model.h5')
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')
kp1, desc1 = get_kp_des(img1)
kp2, desc2 = get_kp_des(img2)
src_pts, dst_pts = get_match(kp1, desc1, kp2, desc2)
H = get_homography(src_pts, dst_pts)
src_pts = np.array(src_pts[:, 0, :])
dst_pts = np.array(dst_pts[:, 0, :])
src_pts = np.expand_dims(src_pts, axis=0)
dst_pts = np.expand_dims(dst_pts, axis=0)
y_pred = model.predict([src_pts, dst_pts])
# H_pred = y_pred.reshape(-1, 3, 3)

# Print the results for each test example
print(f"True homography:{H}")
print(f"Predicted homography:{y_pred}")
print("\n")

# Save the model
model.save('homography_model.h5')
