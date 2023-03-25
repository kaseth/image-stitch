import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import backend as k
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from get_train import data_loader


def euclidean_l2(y_true, y_pred):
    return k.sqrt(k.sum(k.square(y_pred - y_true), axis=-1, keepdims=True))


def homography_net():
    # Feature Extraction Network

    input_shape = (128, 128, 2)
    input_data = tf.keras.Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Regression Network
    flat = Flatten()(x)
    dense = Dense(1024, activation='relu')(flat)
    dense = Dropout(0.5)(dense)
    out = Dense(8)(dense)

    # Define the model
    model = tf.keras.Model(inputs=input_data, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-6),
                  loss='mse')
    return model


# img_input, mat_output = get_train()
# np.savez("training_four_2017", img_input=img_input, mat_output=mat_output)
Ho_model = homography_net()
# Ho_model.load_weights("./checkpoint/w2017.50-0.04432.h5")
Ho_model.summary()

# training = np.load('training_four_2017.npz')
# img_input = training["img_input"]
# mat_output = training["mat_output"]

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoint/w2017.{epoch:02d}-{val_loss:.5f}.h5',
    save_weights_only=True,
    save_freq='epoch'

)
# train_x, test_x, train_y, test_y = train_test_split(img_input, mat_output, test_size=0.1, random_state=10)
train_data_path = './train_npz'
test_data_path = './test_npz'
num_samples = 106200
test_sample = 17700
batch_size = 64
epochs = 50
steps_per_epoch = num_samples / batch_size
validation_steps = test_sample / batch_size
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
Ho_model.fit(data_loader(train_data_path, batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs,
             validation_data=data_loader(test_data_path, batch_size), validation_steps=validation_steps,
             callbacks=[cp_callback])
Ho_model.save("ho_model.h5")
