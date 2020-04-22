# python imports
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# project imports
from . import utils

# third-party imports
import ext.neuron.layers as nrn_layers


def bias_field_augmentation(tensor, bias_field_std=.3, bias_shape_factor=.025):
    """This function applies a bias field to the input tensor. The following steps occur:
    1) a small-size SVF is sampled from a centred normal distribution,
    2) it is resized with trilinear interpolation to image size
    3) it is rescaled to postive values by taking the voxel-wise exponential
    4) it is multiplied to the input tensor.
    :param tensor: input tensor. Expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param bias_field_std: (optional) standard deviation of the normal distribution from which we sample the small SVF.
    :param bias_shape_factor: (optional) ration between the shape of the input tensor and the shape of the sampled SVF.
    :return: a biased tensor
    """

    # reformat tensor and get its shape
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    volume_shape = tensor.get_shape().as_list()[1: -1]
    n_dims = len(volume_shape)

    # sample small field from normal distribution of specified std dev
    small_shape = utils.get_resample_shape(volume_shape, bias_shape_factor, 1)
    tensor_shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
    split_shape = KL.Lambda(lambda x: tf.split(x, [1, n_dims + 1]))(tensor_shape)
    bias_shape = KL.Lambda(lambda x: tf.concat([x, tf.convert_to_tensor(small_shape)], axis=0))(split_shape[0])
    bias_field = KL.Lambda(lambda x: tf.random.normal(x, stddev=bias_field_std))(bias_shape)
    bias_field._keras_shape = tuple(bias_field.get_shape().as_list())

    # resize bias field and take exponential
    bias_field = nrn_layers.Resize(size=volume_shape, interp_method='linear')(bias_field)
    bias_field._keras_shape = tuple(bias_field.get_shape().as_list())
    bias_field = KL.Lambda(lambda x: K.exp(x))(bias_field)

    return KL.multiply([bias_field, tensor])


def min_max_normalisation(tensor):
    """Normalise tensor between 0 and 1"""
    m = KL.Lambda(lambda x: K.min(x))(tensor)
    M = KL.Lambda(lambda x: K.max(x))(tensor)
    return KL.Lambda(lambda x: (x[0] - x[1]) / (x[2] - x[1]))([tensor, m, M])


def gamma_augmentation(tensor, std=0.5):
    """Raise tensor to a power obtained by taking the exp of a value sampled from a gaussian with specified std dev."""
    return KL.Lambda(lambda x: tf.math.pow(x, tf.math.exp(tf.random.normal([1], mean=0, stddev=std))))(tensor)
