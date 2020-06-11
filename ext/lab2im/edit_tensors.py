"""This file contains functions to edit keras/tensorflow tensors.
A lot of them are used in lab2im_model, and we provide them here separately, so they can be re-used easily.
The functions are classified in three categories:
1- blurring functions: They contain functions to create blurring tensors and to apply the obtained kernels:
    -blur_tensor
    -get_gaussian_1d_kernels
    -blur_channel
2- resampling function: function to resample a tensor to a specified resolution.
    -resample_tensor
3- converting label values: these functions only apply to tensors with a limited set of integers as values (typically
label map tensors). It contains:
    -convert_labels
    -reset_label_values_to_zero
4- padding tensor
    -pad_tensor
"""

# python imports
import math
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import tensorflow_probability as tfp

# project imports
from . import utils

# third-party imports
import ext.neuron.layers as nrn_layers


# ------------------------------------------------- blurring functions -------------------------------------------------


def blur_tensor(tensor, list_kernels, n_dims=3):
    """Blur image with masks in list_kernels, if they are not None."""
    for k in list_kernels:
        if k is not None:
            tensor = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1]*n_dims))([tensor, k])
    return tensor


def get_gaussian_1d_kernels(sigma, blurring_range=None):
    """This function builds a list of 1d gaussian blurring kernels.
    The produced tensors are designed to be used with tf.nn.convolution.
    The number of dimensions of the image to blur is assumed to be the length of sigma.
    :param sigma: std deviation of the gaussian kernels to build. Must be a sequence of size n_dims
    (excluding batch and channel dimensions)
    :param blurring_range: if not None, this introduces a randomness in the blurring kernels,
    where sigma is now multiplied by a coefficient dynamically sampled from a uniform distribution with bounds
    [1/blurring_range, blurring_range].
    :return: a list of 1d blurring kernels
    """

    sigma = utils.reformat_to_list(sigma)
    n_dims = len(sigma)

    kernels_list = list()
    for i in range(n_dims):

        if (sigma[i] is None) or (sigma[i] == 0):
            kernels_list.append(None)

        else:
            # build kernel
            if blurring_range is not None:
                random_coef = KL.Lambda(lambda x: tf.random.uniform((1,), 1 / blurring_range, blurring_range))([])
                size = int(math.ceil(2.5 * blurring_range * sigma[i]) / 2)
                kernel = KL.Lambda(lambda x: tfp.distributions.Normal(0., x*sigma[i]).prob(tf.range(start=-size,
                                   limit=size + 1, dtype=tf.float32)))(random_coef)
            else:
                size = int(math.ceil(2.5 * sigma[i]) / 2)
                kernel = KL.Lambda(lambda x: tfp.distributions.Normal(0., sigma[i]).prob(tf.range(start=-size,
                                   limit=size + 1, dtype=tf.float32)))([])
            kernel = KL.Lambda(lambda x: x / tf.reduce_sum(x))(kernel)

            # add dimensions
            for j in range(n_dims):
                if j < i:
                    kernel = KL.Lambda(lambda x: tf.expand_dims(x, 0))(kernel)
                elif j > i:
                    kernel = KL.Lambda(lambda x: tf.expand_dims(x, -1))(kernel)
            kernel = KL.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, -1), -1))(kernel)  # for tf.nn.convolution
            kernels_list.append(kernel)

    return kernels_list


def blur_channel(tensor, mask, kernels_list, n_dims, blur_background=True):
    """Blur a tensor with a list of kernels.
    If blur_background is True, this function enforces a zero background after blurring in 20% of the cases.
    If blur_background is False, this function corrects edge-blurring effects and replaces the zero-backgound by a low
    intensity gaussian noise.
    :param tensor: a input tensor
    :param mask: mask of non-background regions in the input tensor
    :param kernels_list: list of blurring 1d kernels
    :param n_dims: number of dimensions of the initial image (excluding batch and channel dimensions)
    :param blur_background: whether to correct for edge-blurring effects
    :return: blurred tensor with background augmentation
    """

    # blur image
    tensor = blur_tensor(tensor, kernels_list, n_dims)

    if blur_background:  # background already blurred with the rest of the image

        # enforce zero background in 20% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.8))([])
        tensor = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                              KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'),
                                                                           x[0], tf.zeros_like(x[0])))([y[1], y[2]]),
                                              y[1]))([rand, tensor, mask])

    else:  # correct for edge blurring effects

        # blur mask and correct edge blurring effects
        blurred_mask = blur_tensor(mask, kernels_list, n_dims)
        tensor = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([tensor, blurred_mask])

        # replace zero background by low intensity background in 50% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])
        bckgd_mean = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 20))([])
        bckgd_std = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 10))([])
        bckgd_mean = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                  KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                  y[1]))([rand, bckgd_mean])
        bckgd_std = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                 KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                 y[1]))([rand, bckgd_std])
        background = KL.Lambda(lambda x: x[1] + x[2]*tf.random.normal(tf.shape(x[0])))([tensor, bckgd_mean, bckgd_std])
        background_kernels = get_gaussian_1d_kernels(sigma=[1]*3)
        background = blur_tensor(background, background_kernels, n_dims)
        tensor = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]))([tensor, mask, background])

    return tensor


# ------------------------------------------------ resampling functions ------------------------------------------------

def resample_tensor(tensor,
                    resample_shape,
                    interp_method='linear',
                    subsample_res=None,
                    volume_res=None,
                    subsample_interp_method='nearest',
                    n_dims=3):
    """This function resamples a volume to resample_shape. It does not apply any pre-filtering.
    A prior downsampling step can be added if subsample_res is specified. In this case, volume_res should also be
    specified, in order to calculate the downsampling ratio.
    :param tensor: tensor
    :param resample_shape: list or numpy array of size (n_dims,)
    :param interp_method: interpolation method for resampling, 'linear' or 'nearest'
    :param subsample_res: if not None, this triggers a downsampling of the volume, prior to the resampling step.
    list or numpy array of size (n_dims,).
    :param volume_res: if subsample_res is not None, this should be provided to compute downsampling ratio.
     list or numpy array of size (n_dims,).
    :param subsample_interp_method: interpolation method for downsampling, 'linear' or 'nearest'
    :param n_dims: number of dimensions of the initial image (excluding batch and channel dimensions)
    :return: resampled volume
    """

    # downsample image
    downsample_shape = None
    tensor_shape = tensor.get_shape().as_list()[1:-1]
    if subsample_res is not None:
        if subsample_res.tolist() != volume_res.tolist():

            # get shape at which we downsample
            assert volume_res is not None, 'if subsanple_res is specified, so should atlas_res be.'
            downsample_factor = [volume_res[i] / subsample_res[i] for i in range(n_dims)]
            downsample_shape = [int(tensor_shape[i] * downsample_factor[i]) for i in range(n_dims)]

            # downsample volume
            tensor._keras_shape = tuple(tensor.get_shape().as_list())
            tensor = nrn_layers.Resize(size=downsample_shape, interp_method=subsample_interp_method)(tensor)

    # resample image at target resolution
    if resample_shape != downsample_shape:
        tensor._keras_shape = tuple(tensor.get_shape().as_list())
        tensor = nrn_layers.Resize(size=resample_shape, interp_method=interp_method)(tensor)

    return tensor


# ------------------------------------------------ convert label values ------------------------------------------------

def convert_labels(label_map, labels_list):
    """Change all labels in label_map by the values in labels_list"""
    return KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(labels_list, dtype='int32'),
                                         tf.cast(x, dtype='int32')))(label_map)


def reset_label_values_to_zero(label_map, labels_to_reset):
    """Reset to zero all occurences in label_map of the values contained in labels_to_remove.
    :param label_map: tensor
    :param labels_to_reset: list of values to reset to zero
    """
    for lab in labels_to_reset:
        label_map = KL.Lambda(lambda x: tf.where(tf.equal(tf.cast(x, dtype='int32'),
                                                          tf.cast(tf.convert_to_tensor(lab), dtype='int32')),
                                                 tf.zeros_like(x, dtype='int32'),
                                                 tf.cast(x, dtype='int32')))(label_map)
    return label_map


# ---------------------------------------------------- pad tensors -----------------------------------------------------

def pad_tensor(tensor, padding_shape=None, pad_value=0):
    """Pad tensor to specified shape.
    :param tensor: tensor to pad
    :param padding_shape: shape of the returned padded tensor. Can be a list or a numy 1d array, of the same length as
    the numbe of dimensions of the tensor (including batch and channel dimensions).
    :param pad_value: value by which to pad the tensor. Default is 0.
    """

    # get shapes and padding margins
    tensor_shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
    padding_shape = KL.Lambda(lambda x: tf.math.maximum(x, tf.convert_to_tensor(padding_shape, tf.int32)))(tensor_shape)

    # build padding margins
    min_margins = KL.Lambda(lambda x: tf.cast((x[0] - x[1]) / 2, tf.int32))([padding_shape, tensor_shape])
    max_margins = KL.Lambda(lambda x: (x[0] - x[1]) - x[2])([padding_shape, tensor_shape, min_margins])
    margins = KL.Lambda(lambda x: tf.stack([x[0], tf.cast(x[1], tf.int32)], axis=-1))([min_margins, max_margins])

    # pad tensor
    padded_tensor = KL.Lambda(lambda x: tf.pad(x[0], x[1], 'CONSTANT', constant_values=pad_value))([tensor, margins])
    return padded_tensor
