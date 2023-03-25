import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the two images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=img1.shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
x_train = np.array([img1])
y_train = np.array([img2])
model.fit(x_train, y_train, epochs=10)

# Use the model to find corresponding features
matches = []
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        feature1 = np.array([img1[i][j]])
        feature2 = model.predict(np.array([feature1]))
        if np.array_equal(feature2, np.array([img2[i][j]])):
            matches.append((i, j))

# Create the homography matrix
if len(matches) >= 4:
    src_pts = np.float32([ [img1[i][j][0], img1[i][j][1]] for (i, j) in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ [img2[i][j][0], img2[i][j][1]] for (i, j) in matches ]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
else:
    print("Not enough matches")
