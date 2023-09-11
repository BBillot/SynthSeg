from tensorflow import keras
from typing import Tuple


def unet(input_shape: Tuple[int, ...], num_classes):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv3D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)

    for filters in [64, 128, 256, 512]:
        x = keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling3D(pool_size=2)(x)

    for filters in [512, 256, 128, 64, 32]:
        x = keras.layers.Conv3DTranspose(filters, 3, padding="same", activation="relu")(
            x
        )
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv3DTranspose(filters, 3, padding="same", activation="relu")(
            x
        )
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling3D(2)(x)

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv3D(
        num_classes, 3, activation="softmax", padding="same", name="unet_likelihood"
    )(x)

    # Define the model
    model = keras.Model(inputs, outputs)

    return model
