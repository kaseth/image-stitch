import tensorflow as tf
import numpy as np
from get_train import data_loader


def homography_net(input_shape=(128, 128, 2)):
    input_layer = tf.keras.Input(shape=input_shape)

    # Convolutional layers
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Fully connected layers
    flatten = tf.keras.layers.Flatten()(x)
    fc1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    fc2 = tf.keras.layers.Dropout(0.5)(fc1)
    out = tf.keras.layers.Dense(8)(fc2)

    # Create model
    h_model = tf.keras.models.Model(inputs=input_layer, outputs=out)
    h_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-6), loss="mse")

    return h_model


model = homography_net()
model.summary()
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./deep_check/w.{epoch:02d}-{val_loss:.2f}.h5',
    save_weights_only=True,
    save_freq='epoch'

)
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

train_data_path = './train_npz'
test_data_path = './test_npz'
num_samples = 106200
test_sample = 17700
batch_size = 64
epochs = 50
steps_per_epoch = num_samples / batch_size
validation_steps = test_sample / batch_size


model.fit(data_loader(train_data_path, batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs,
          validation_data=data_loader(test_data_path, batch_size), validation_steps=validation_steps,
          callbacks=[cp_callback])
model.save("model.h5")
