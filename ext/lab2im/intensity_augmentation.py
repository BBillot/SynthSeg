# python imports
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# third-party imports
import ext.neuron.layers as nrn_layers


def bias_field_augmentation(tensor, bias_field, n_dims=3):
    """This function takes a bias_field as input, under the form of a small grid.
    The bias field is first resampled to image size, and rescaled to postive values by taking its exponential.
    The bias field is applied by multiplying it with the image."""

    # resize bias field and take exponential
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    bias_field = nrn_layers.Resize(size=image_shape, interp_method='linear')(bias_field)
    bias_field = KL.Lambda(lambda x: K.exp(x))(bias_field)

    # apply bias_field
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    bias_field._keras_shape = tuple(bias_field.get_shape().as_list())
    return KL.multiply([bias_field, tensor])


def min_max_normalisation(tensor):
    """Normalise tensor between 0 and 1"""
    m = KL.Lambda(lambda x: K.min(x))(tensor)
    M = KL.Lambda(lambda x: K.max(x))(tensor)
    return KL.Lambda(lambda x: (x[0] - x[1]) / (x[2] - x[1]))([tensor, m, M])


def gamma_augmentation(tensor, std=0.5):
    """Raise tensor to a power obtained by taking the exp of a value sampled from a gaussian with specified std dev."""
    return KL.Lambda(lambda x: tf.math.pow(x, tf.math.exp(tf.random.normal([1], mean=0, stddev=std))))(tensor)
