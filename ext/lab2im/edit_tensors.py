"""

This file contains functions to handle keras/tensorflow tensors.
    - blurring_sigma_for_downsampling
    - gaussian_kernel
    - resample_tensor
    - expand_dims


If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.

"""


# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from itertools import combinations

# project imports
from ext.lab2im import utils

# third-party imports
import ext.neuron.layers as nrn_layers
from ext.neuron.utils import volshape_to_meshgrid


def blurring_sigma_for_downsampling(current_res, downsample_res, mult_coef=None, thickness=None):
    """Compute standard deviations of 1d gaussian masks for image blurring before downsampling.
    :param downsample_res: resolution to downsample to. Can be a 1d numpy array or list, or a tensor.
    :param current_res: resolution of the volume before downsampling.
    Can be a 1d numpy array or list or tensor of the same length as downsample res.
    :param mult_coef: (optional) multiplicative coefficient for the blurring kernel. Default is 0.75.
    :param thickness: (optional) slice thickness in each dimension. Must be the same type as downsample_res.
    :return: standard deviation of the blurring masks given as the same type as downsample_res (list or tensor).
    """

    if not tf.is_tensor(downsample_res):

        # get blurring resolution (min between downsample_res and thickness)
        current_res = np.array(current_res)
        downsample_res = np.array(downsample_res)
        if thickness is not None:
            downsample_res = np.minimum(downsample_res, np.array(thickness))

        # get std deviation for blurring kernels
        if mult_coef is None:
            sigma = 0.75 * downsample_res / current_res
            sigma[downsample_res == current_res] = 0.5
        else:
            sigma = mult_coef * downsample_res / current_res
        sigma[downsample_res == 0] = 0

    else:

        # reformat data resolution at which we blur
        if thickness is not None:
            down_res = KL.Lambda(lambda x: tf.math.minimum(x[0], x[1]))([downsample_res, thickness])
        else:
            down_res = downsample_res

        # get std deviation for blurring kernels
        if mult_coef is None:
            sigma = KL.Lambda(lambda x: tf.where(tf.math.equal(x, tf.convert_to_tensor(current_res, dtype='float32')),
                              0.5, 0.75 * x / tf.convert_to_tensor(current_res, dtype='float32')))(down_res)
        else:
            sigma = KL.Lambda(lambda x: mult_coef * x / tf.convert_to_tensor(current_res, dtype='float32'))(down_res)
        sigma = KL.Lambda(lambda x: tf.where(tf.math.equal(x[0], 0.), 0., x[1]))([down_res, sigma])

    return sigma


def gaussian_kernel(sigma, max_sigma=None, blur_range=None, separable=True):
    """Build gaussian kernels of the specified standard deviation. The outputs are given as tensorflow tensors.
    :param sigma: standard deviation of the tensors. Can be given as a list/numpy array or as tensors. In each case,
    sigma must have the same length as the number of dimensions of the volume that will be blurred with the output
    tensors (e.g. sigma must have 3 values for 3D volumes).
    :param max_sigma:
    :param blur_range:
    :param separable:
    :return:
    """
    # convert sigma into a tensor
    if not tf.is_tensor(sigma):
        sigma_tens = tf.convert_to_tensor(utils.reformat_to_list(sigma), dtype='float32')
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    shape = sigma_tens.get_shape().as_list()

    # get n_dims and batchsize
    if shape[0] is not None:
        n_dims = shape[0]
        batchsize = None
    else:
        n_dims = shape[1]
        batchsize = tf.split(tf.shape(sigma_tens), [1, -1])[0]

    # reformat max_sigma
    if max_sigma is not None:  # dynamic blurring
        max_sigma = np.array(utils.reformat_to_list(max_sigma, length=n_dims))
    else:  # sigma is fixed
        max_sigma = np.array(utils.reformat_to_list(sigma, length=n_dims))

    # randomise the burring std dev and/or split it between dimensions
    if blur_range is not None:
        if blur_range != 1:
            sigma_tens = sigma_tens * tf.random.uniform(tf.shape(sigma_tens), minval=1 / blur_range, maxval=blur_range)

    # get size of blurring kernels
    windowsize = np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1

    if separable:

        split_sigma = tf.split(sigma_tens, [1] * n_dims, axis=-1)

        kernels = list()
        comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])
        for (i, wsize) in enumerate(windowsize):

            if wsize > 1:

                # build meshgrid and replicate it along batch dim if dynamic blurring
                locations = tf.cast(tf.range(0, wsize), 'float32') - (wsize - 1) / 2
                if batchsize is not None:
                    locations = tf.tile(tf.expand_dims(locations, axis=0),
                                        tf.concat([batchsize, tf.ones(tf.shape(tf.shape(locations)), dtype='int32')],
                                                  axis=0))
                    comb[i] += 1

                # compute gaussians
                exp_term = -K.square(locations) / (2 * split_sigma[i] ** 2)
                g = tf.exp(exp_term - tf.math.log(np.sqrt(2 * np.pi) * split_sigma[i]))
                g = g / tf.reduce_sum(g)

                for axis in comb[i]:
                    g = tf.expand_dims(g, axis=axis)
                kernels.append(tf.expand_dims(tf.expand_dims(g, -1), -1))

            else:
                kernels.append(None)

    else:

        # build meshgrid
        mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
        diff = tf.stack([mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)

        # replicate meshgrid to batch size and reshape sigma_tens
        if batchsize is not None:
            diff = tf.tile(tf.expand_dims(diff, axis=0),
                           tf.concat([batchsize, tf.ones(tf.shape(tf.shape(diff)), dtype='int32')], axis=0))
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=1)
        else:
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=0)

        # compute gaussians
        sigma_is_0 = tf.equal(sigma_tens, 0)
        exp_term = -K.square(diff) / (2 * tf.where(sigma_is_0, tf.ones_like(sigma_tens), sigma_tens)**2)
        norms = exp_term - tf.math.log(tf.where(sigma_is_0, tf.ones_like(sigma_tens), np.sqrt(2 * np.pi) * sigma_tens))
        kernels = K.sum(norms, -1)
        kernels = tf.exp(kernels)
        kernels /= tf.reduce_sum(kernels)
        kernels = tf.expand_dims(tf.expand_dims(kernels, -1), -1)

    return kernels


def sobel_kernels(n_dims):
    """Returns sobel kernels to compute spatial derivative on image of n dimensions."""

    in_dir = tf.convert_to_tensor([1, 0, -1], dtype='float32')
    orthogonal_dir = tf.convert_to_tensor([1, 2, 1], dtype='float32')
    comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])

    list_kernels = list()
    for dim in range(n_dims):

        sublist_kernels = list()
        for axis in range(n_dims):

            kernel = in_dir if axis == dim else orthogonal_dir
            for i in comb[axis]:
                kernel = tf.expand_dims(kernel, axis=i)
            sublist_kernels.append(tf.expand_dims(tf.expand_dims(kernel, -1), -1))

        list_kernels.append(sublist_kernels)

    return list_kernels


def unit_kernel(dist_threshold, n_dims, max_dist_threshold=None):
    """Build kernel with values of 1 for voxel at a distance < dist_threshold from the center, and 0 otherwise.
    The outputs are given as tensorflow tensors.
    :param dist_threshold: maximum distance from the center until voxel will have a value of 1. Can be a tensor of size
    (batch_size, 1), or a float.
    :param n_dims: dimension of the kernel to return (excluding batch and channel dimensions).
    :param max_dist_threshold: if distance_threshold is a tensor, max_dist_threshold must be given. It represents the
    maximum value that will be passed to dist_threshold. Must be a float.
    """

    # convert dist_threshold into a tensor
    if not tf.is_tensor(dist_threshold):
        dist_threshold_tens = tf.convert_to_tensor(utils.reformat_to_list(dist_threshold), dtype='float32')
    else:
        assert max_dist_threshold is not None, 'max_sigma must be provided when dist_threshold is given as a tensor'
        dist_threshold_tens = tf.cast(dist_threshold, 'float32')
    shape = dist_threshold_tens.get_shape().as_list()

    # get batchsize
    batchsize = None if shape[0] is not None else tf.split(tf.shape(dist_threshold_tens), [1, -1])[0]

    # set max_dist_threshold into an array
    if max_dist_threshold is None:  # dist_threshold is fixed (i.e. dist_threshold will not change at each mini-batch)
        max_dist_threshold = dist_threshold

    # get size of blurring kernels
    windowsize = np.array([max_dist_threshold * 2 + 1]*n_dims, dtype='int32')

    # build tensor representing the distance from the centre
    mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
    dist = tf.stack([mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)
    dist = tf.sqrt(tf.reduce_sum(tf.square(dist), axis=-1))

    # replicate distance to batch size and reshape sigma_tens
    if batchsize is not None:
        dist = tf.tile(tf.expand_dims(dist, axis=0),
                       tf.concat([batchsize, tf.ones(tf.shape(tf.shape(dist)), dtype='int32')], axis=0))
        for i in range(n_dims - 1):
            dist_threshold_tens = tf.expand_dims(dist_threshold_tens, axis=1)
    else:
        for i in range(n_dims - 1):
            dist_threshold_tens = tf.expand_dims(dist_threshold_tens, axis=0)

    # build final kernel by thresholding distance tensor
    kernel = tf.where(tf.less_equal(dist, dist_threshold_tens), tf.ones_like(dist), tf.zeros_like(dist))
    kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)

    return kernel


def resample_tensor(tensor,
                    resample_shape,
                    interp_method='linear',
                    subsample_res=None,
                    volume_res=None,
                    build_reliability_map=False):
    """This function resamples a volume to resample_shape. It does not apply any pre-filtering.
    A prior downsampling step can be added if subsample_res is specified. In this case, volume_res should also be
    specified, in order to calculate the downsampling ratio. A reliability map can also be returned to indicate which
    slices were interpolated during resampling from the downsampled to final tensor.
    :param tensor: tensor
    :param resample_shape: list or numpy array of size (n_dims,)
    :param interp_method: (optional) interpolation method for resampling, 'linear' (default) or 'nearest'
    :param subsample_res: (optional) if not None, this triggers a downsampling of the volume, prior to the resampling
    step. List or numpy array of size (n_dims,). Default si None.
    :param volume_res: (optional) if subsample_res is not None, this should be provided to compute downsampling ratio.
    list or numpy array of size (n_dims,). Default is None.
    :param build_reliability_map: whether to return reliability map along with the resampled tensor. This map indicates
    which slices of the resampled tensor are interpolated (0=interpolated, 1=real slice, in between=degree of realness).
    :return: resampled volume, with reliability map if necessary.
    """

    # reformat resolutions to lists
    subsample_res = utils.reformat_to_list(subsample_res)
    volume_res = utils.reformat_to_list(volume_res)
    n_dims = len(resample_shape)

    # downsample image
    tensor_shape = tensor.get_shape().as_list()[1:-1]
    downsample_shape = tensor_shape  # will be modified if we actually downsample

    if subsample_res is not None:
        assert volume_res is not None, 'volume_res must be given when providing a subsampling resolution.'
        assert len(subsample_res) == len(volume_res), 'subsample_res and volume_res must have the same length, ' \
                                                      'had {0}, and {1}'.format(len(subsample_res), len(volume_res))
        if subsample_res != volume_res:

            # get shape at which we downsample
            downsample_shape = [int(tensor_shape[i] * volume_res[i] / subsample_res[i]) for i in range(n_dims)]

            # downsample volume
            tensor._keras_shape = tuple(tensor.get_shape().as_list())
            tensor = nrn_layers.Resize(size=downsample_shape, interp_method='nearest')(tensor)

    # resample image at target resolution
    if resample_shape != downsample_shape:  # if we didn't downsample downsample_shape = tensor_shape
        tensor._keras_shape = tuple(tensor.get_shape().as_list())
        tensor = nrn_layers.Resize(size=resample_shape, interp_method=interp_method)(tensor)

    # compute reliability maps if necessary and return results
    if build_reliability_map:

        # compute maps only if we downsampled
        if downsample_shape != tensor_shape:

            # compute upsampling factors
            upsampling_factors = np.array(resample_shape) / np.array(downsample_shape)

            # build reliability map
            reliability_map = 1
            for i in range(n_dims):
                loc_float = np.arange(0, resample_shape[i], upsampling_factors[i])
                loc_floor = np.int32(np.floor(loc_float))
                loc_ceil = np.int32(np.clip(loc_floor + 1, 0, resample_shape[i] - 1))
                tmp_reliability_map = np.zeros(resample_shape[i])
                tmp_reliability_map[loc_floor] = 1 - (loc_float - loc_floor)
                tmp_reliability_map[loc_ceil] = tmp_reliability_map[loc_ceil] + (loc_float - loc_floor)
                shape = [1, 1, 1]
                shape[i] = resample_shape[i]
                reliability_map = reliability_map * np.reshape(tmp_reliability_map, shape)
            shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
            mask = KL.Lambda(lambda x: tf.reshape(tf.convert_to_tensor(reliability_map, dtype='float32'),
                                                  shape=x))(shape)

        # otherwise just return an all-one tensor
        else:
            mask = KL.Lambda(lambda x: tf.ones_like(x))(tensor)

        return tensor, mask

    else:
        return tensor


def expand_dims(tensor, axis=0):
    """Expand the dimensions of the input tensor along the provided axes (given as an integer or a list)."""
    axis = utils.reformat_to_list(axis)
    for ax in axis:
        tensor = tf.expand_dims(tensor, axis=ax)
    return tensor
