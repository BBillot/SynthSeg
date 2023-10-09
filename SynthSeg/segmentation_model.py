import tensorflow as tf
from typing import Tuple


def unet(
    input_shape: Tuple[int, ...],
    n_labels: int,
    unet_feat_count: int = 24,
    conv_size: int = 3,
    n_levels: int = 5,
    nb_conv_per_level: int = 2,
    activation: str = "elu",
):
    inputs = tf.keras.Input(shape=input_shape)

    skip_connections = []

    x = inputs

    # encoder
    for level in range(n_levels):
        lvl_feats = unet_feat_count * 2**level
        for _ in range(nb_conv_per_level):
            x = tf.keras.layers.Conv3D(
                lvl_feats, conv_size, padding="same", activation=activation
            )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if level != n_levels - 1:
            skip_connections.append(x)
            x = tf.keras.layers.MaxPooling3D()(x)

    # decoder
    for level in reversed(range(n_levels-1)):
        x = tf.keras.layers.UpSampling3D()(x)
        x = tf.keras.layers.Concatenate()([x, skip_connections[level]])

        lvl_feats = unet_feat_count * 2**level
        for _ in range(nb_conv_per_level):
            x = tf.keras.layers.Conv3D(
                lvl_feats, conv_size, padding="same", activation=activation
            )(x)

        x = tf.keras.layers.BatchNormalization()(x)

    # Add a per-pixel likelihood layer
    x = tf.keras.layers.Conv3D(
        n_labels,
        1,
        activation=None,
        name="unet_likelihood",
    )(x)

    # Add a per-pixel prediction layer
    outputs = tf.keras.layers.Softmax()(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)

    return model
