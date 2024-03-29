"""
This file regroups several custom keras layers used in the generation model:
    - RandomSpatialDeformation,
    - RandomCrop,
    - RandomFlip,
    - SampleConditionalGMM,
    - SampleResolution,
    - GaussianBlur,
    - DynamicGaussianBlur,
    - MimicAcquisition,
    - BiasFieldCorruption,
    - IntensityAugmentation,
    - DiceLoss,
    - WeightedL2Loss,
    - ResetValuesToZero,
    - ConvertLabels,
    - PadAroundCentre,
    - MaskEdges
    - ImageGradients
    - RandomDilationErosion


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
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

# project imports
from ext.lab2im import utils
from ext.lab2im import edit_tensors as l2i_et

# third-party imports
from ext.neuron import utils as nrn_utils
import ext.neuron.layers as nrn_layers


class RandomSpatialDeformation(Layer):
    """This layer spatially deforms one or several tensors with a combination of affine and elastic transformations.
    The input tensors are expected to have the same shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    The non-linear deformation is obtained by:
    1) a small-size SVF is sampled from a centred normal distribution of random standard deviation.
    2) it is resized with trilinear interpolation to half the shape of the input tensor
    3) it is integrated to obtain a diffeomorphic transformation
    4) finally, it is resized (again with trilinear interpolation) to full image size
    :param scaling_bounds: (optional) range of the random scaling to apply. The scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    [1-scaling_bounds, 1+scaling_bounds] for each dimension.
    2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
    (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
    3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
     of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    4) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
    and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
    Default is rotation_bounds = 15.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
    :param enable_90_rotations: (optional) whether to rotate the input by a random angle chosen in {0, 90, 180, 270}.
    This is done regardless of the value of rotation_bounds. If true, a different value is sampled for each dimension.
    :param nonlin_std: (optional) maximum value of the standard deviation of the normal distribution from which we
    sample the small-size SVF. Set to 0 if you wish to completely turn the elastic deformation off.
    :param nonlin_scale: (optional) if nonlin_std is not False, factor between the shapes of the input tensor
    and the shape of the input non-linear tensor.
    :param inter_method: (optional) interpolation method when deforming the input tensor. Can be 'linear', or 'nearest'
    :param prob_deform: (optional) probability to apply spatial deformation
    """

    def __init__(self,
                 scaling_bounds=0.15,
                 rotation_bounds=10,
                 shearing_bounds=0.02,
                 translation_bounds=False,
                 enable_90_rotations=False,
                 nonlin_std=4.,
                 nonlin_scale=.0625,
                 inter_method='linear',
                 prob_deform=1,
                 **kwargs):

        # shape attributes
        self.n_inputs = 1
        self.inshape = None
        self.n_dims = None
        self.small_shape = None

        # deformation attributes
        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.enable_90_rotations = enable_90_rotations
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale

        # boolean attributes
        self.apply_affine_trans = (self.scaling_bounds is not False) | (self.rotation_bounds is not False) | \
                                  (self.shearing_bounds is not False) | (self.translation_bounds is not False) | \
                                  self.enable_90_rotations
        self.apply_elastic_trans = self.nonlin_std > 0
        self.prob_deform = prob_deform

        # interpolation methods
        self.inter_method = inter_method

        super(RandomSpatialDeformation, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["scaling_bounds"] = self.scaling_bounds
        config["rotation_bounds"] = self.rotation_bounds
        config["shearing_bounds"] = self.shearing_bounds
        config["translation_bounds"] = self.translation_bounds
        config["enable_90_rotations"] = self.enable_90_rotations
        config["nonlin_std"] = self.nonlin_std
        config["nonlin_scale"] = self.nonlin_scale
        config["inter_method"] = self.inter_method
        config["prob_deform"] = self.prob_deform
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            inputshape = [input_shape]
        else:
            self.n_inputs = len(input_shape)
            inputshape = input_shape
        self.inshape = inputshape[0][1:]
        self.n_dims = len(self.inshape) - 1

        if self.apply_elastic_trans:
            self.small_shape = utils.get_resample_shape(self.inshape[:self.n_dims],
                                                        self.nonlin_scale, self.n_dims)
        else:
            self.small_shape = None

        self.inter_method = utils.reformat_to_list(self.inter_method, length=self.n_inputs, dtype='str')

        self.built = True
        super(RandomSpatialDeformation, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # reformat inputs and get its shape
        if self.n_inputs < 2:
            inputs = [inputs]
        types = [v.dtype for v in inputs]
        inputs = [tf.cast(v, dtype='float32') for v in inputs]
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]

        # initialise list of transforms to operate
        list_trans = list()

        # add affine deformation to inputs list
        if self.apply_affine_trans:
            affine_trans = utils.sample_affine_transform(batchsize,
                                                         self.n_dims,
                                                         self.rotation_bounds,
                                                         self.scaling_bounds,
                                                         self.shearing_bounds,
                                                         self.translation_bounds,
                                                         self.enable_90_rotations)
            list_trans.append(affine_trans)

        # prepare non-linear deformation field and add it to inputs list
        if self.apply_elastic_trans:

            # sample small field from normal distribution of specified std dev
            trans_shape = tf.concat([batchsize, tf.convert_to_tensor(self.small_shape, dtype='int32')], axis=0)
            trans_std = tf.random.uniform((1, 1), maxval=self.nonlin_std)
            elastic_trans = tf.random.normal(trans_shape, stddev=trans_std)

            # reshape this field to half size (for smoother SVF), integrate it, and reshape to full image size
            resize_shape = [max(int(self.inshape[i] / 2), self.small_shape[i]) for i in range(self.n_dims)]
            elastic_trans = nrn_layers.Resize(size=resize_shape, interp_method='linear')(elastic_trans)
            elastic_trans = nrn_layers.VecInt()(elastic_trans)
            elastic_trans = nrn_layers.Resize(size=self.inshape[:self.n_dims], interp_method='linear')(elastic_trans)
            list_trans.append(elastic_trans)

        # apply deformations and return tensors with correct dtype
        if self.apply_affine_trans | self.apply_elastic_trans:
            if self.prob_deform == 1:
                inputs = [nrn_layers.SpatialTransformer(m)([v] + list_trans) for (m, v) in
                          zip(self.inter_method, inputs)]
            else:
                rand_trans = tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_deform))
                inputs = [K.switch(rand_trans, nrn_layers.SpatialTransformer(m)([v] + list_trans), v)
                          for (m, v) in zip(self.inter_method, inputs)]
        if self.n_inputs < 2:
            return tf.cast(inputs[0], types[0])
        else:
            return [tf.cast(v, t) for (t, v) in zip(types, inputs)]


class RandomCrop(Layer):
    """Randomly crop all input tensors to a given shape. This cropping is applied to all channels.
    The input tensors are expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param crop_shape: list with cropping shape in each dimension (excluding batch and channel dimension)

    example:
    if input is a tensor of shape [batchsize, 160, 160, 160, 3],
    output = RandomCrop(crop_shape=[96, 128, 96])(input)
    will yield an output of shape [batchsize, 96, 128, 96, 3] that is obtained by cropping with randomly selected
    cropping indices.
    """

    def __init__(self, crop_shape, **kwargs):

        self.several_inputs = True
        self.crop_max_val = None
        self.crop_shape = crop_shape
        self.n_dims = len(crop_shape)
        self.list_n_channels = None
        super(RandomCrop, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["crop_shape"] = self.crop_shape
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            self.several_inputs = False
            inputshape = [input_shape]
        else:
            inputshape = input_shape
        self.crop_max_val = np.array(np.array(inputshape[0][1:self.n_dims + 1])) - np.array(self.crop_shape)
        self.list_n_channels = [i[-1] for i in inputshape]
        self.built = True
        super(RandomCrop, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # if one input only is provided, performs the cropping directly
        if not self.several_inputs:
            return tf.map_fn(self._single_slice, inputs, dtype=inputs.dtype)

        # otherwise we concatenate all inputs before cropping, so that they are all cropped at the same location
        else:
            types = [v.dtype for v in inputs]
            inputs = tf.concat([tf.cast(v, 'float32') for v in inputs], axis=-1)
            inputs = tf.map_fn(self._single_slice, inputs, dtype=tf.float32)
            inputs = tf.split(inputs, self.list_n_channels, axis=-1)
            return [tf.cast(v, t) for (t, v) in zip(types, inputs)]

    def _single_slice(self, vol):
        crop_idx = tf.cast(tf.random.uniform([self.n_dims], 0, np.array(self.crop_max_val), 'float32'), dtype='int32')
        crop_idx = tf.concat([crop_idx, tf.zeros([1], dtype='int32')], axis=0)
        crop_size = tf.convert_to_tensor(self.crop_shape + [-1], dtype='int32')
        return tf.slice(vol, begin=crop_idx, size=crop_size)

    def compute_output_shape(self, input_shape):
        output_shape = [tuple([None] + self.crop_shape + [v]) for v in self.list_n_channels]
        return output_shape if self.several_inputs else output_shape[0]


class RandomFlip(Layer):
    """This layer randomly flips the input tensor along the specified axes with a specified probability.
    It can also take multiple tensors as inputs (if they have the same shape). The same flips will be applied to all
    input tensors. These are expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    If specified, this layer can also swap corresponding values. This is especially useful when flipping label maps
    with different labels for right/left structures, such that the flipped label maps keep a consistent labelling.
    :param axis: integer, or list of integers specifying the dimensions along which to flip.
    If a list, the input tensors can be flipped simultaneously in several directions. The values in flip_axis exclude
    the batch dimension (e.g. 0 will flip the tensor along the first axis after the batch dimension).
    Default is None, where the tensors can be flipped along all axes (except batch and channel axes).
    :param swap_labels: boolean to specify whether to swap the values of each input. Values are only swapped if an odd
    number of flips is applied.
    Can also be a list if several tensors are given as input.
    All the inputs for which the values need to be swapped must be int32 or int64.
    :param label_list: if swap_labels is True, list of all labels contained in labels. Must be ordered as follows, first
     the neutral labels (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: if swap_labels is True, number of non-sided labels
    :param prob: probability to flip along each specified axis

    example 1:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip()(input) will randomly flip input along one of the 1st, 2nd, or 3rd axis (i.e. those with shape
    10, 100, 200).

    example 2:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip(flip_axis=1)(input) will randomly flip input along the 3rd axis (with shape 100), i.e. the axis
    with index 1 if we don't count the batch axis.

    example 3:
    input = tf.convert_to_tensor(np.array([[1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 0, 0, 0]]))
    label_list = np.array([0, 1, 2])
    n_neutral_labels = 1
    output = RandomFlip(flip_axis=1, swap_labels=True, label_list=label_list, n_neutral_labels=n_neutral_labels)(input)
    where output will either be equal to input (bear in mind the flipping occurs with a 0.5 probability), or:
    output = [[0, 0, 0, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 2]]
    Note that the input must have a dtype int32 or int64 for its values to be swapped, otherwise an error will be raised

    example 4:
    if labels is the same as in the input of example 3, and image is a float32 image, then we can swap consistently both
    the labels and the image with:
    labels, image = RandomFlip(flip_axis=1, swap_labels=[True, False], label_list=label_list,
                               n_neutral_labels=n_neutral_labels)([labels, image]])
    Note that the labels must have a dtype int32 or int64 to be swapped, otherwise an error will be raised.
    This doesn't concern the image input, as its values are not swapped.
    """

    def __init__(self, axis=None, swap_labels=False, label_list=None, n_neutral_labels=None, prob=0.5, **kwargs):

        # shape attributes
        self.several_inputs = True
        self.n_dims = None
        self.list_n_channels = None

        # axis along which to flip
        self.axis = utils.reformat_to_list(axis)
        self.flip_axes = None

        # whether to swap labels, and corresponding label list
        self.swap_labels = utils.reformat_to_list(swap_labels)
        self.label_list = label_list
        self.n_neutral_labels = n_neutral_labels
        self.swap_lut = None

        self.prob = prob

        super(RandomFlip, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["swap_labels"] = self.swap_labels
        config["label_list"] = self.label_list
        config["n_neutral_labels"] = self.n_neutral_labels
        config["prob"] = self.prob
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            self.several_inputs = False
            inputshape = [input_shape]
        else:
            inputshape = input_shape
        self.n_dims = len(inputshape[0][1:-1])
        self.list_n_channels = [i[-1] for i in inputshape]
        self.swap_labels = utils.reformat_to_list(self.swap_labels, length=len(inputshape))
        self.flip_axes = np.arange(self.n_dims).tolist() if self.axis is None else self.axis

        # create label list with swapped labels
        if any(self.swap_labels):
            assert (self.label_list is not None) & (self.n_neutral_labels is not None), \
                'please provide a label_list, and n_neutral_labels when swapping the values of at least one input'
            n_labels = len(self.label_list)
            if self.n_neutral_labels == n_labels:
                self.swap_labels = [False] * len(self.swap_labels)
            else:
                rl_split = np.split(self.label_list, [self.n_neutral_labels,
                                                      self.n_neutral_labels + int((n_labels-self.n_neutral_labels)/2)])
                label_list_swap = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
                swap_lut = utils.get_mapping_lut(self.label_list, label_list_swap)
                self.swap_lut = tf.convert_to_tensor(swap_lut, dtype='int32')

        self.built = True
        super(RandomFlip, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # convert inputs to list, and get each input type
        inputs = [inputs] if not self.several_inputs else inputs
        types = [v.dtype for v in inputs]

        # store whether to flip along each specified dimension
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]
        size = tf.concat([batchsize, len(self.flip_axes) * tf.ones(1, dtype='int32')], axis=0)
        rand_flip = K.less(tf.random.uniform(size, 0, 1), self.prob)

        # swap right/left labels if we apply an odd number of flips
        odd = tf.math.floormod(tf.reduce_sum(tf.cast(rand_flip, 'int32'), -1, keepdims=True), 2) != 0
        swapped_inputs = list()
        for i in range(len(inputs)):
            if self.swap_labels[i]:
                swapped_inputs.append(tf.map_fn(self._single_swap, [inputs[i], odd], dtype=types[i]))
            else:
                swapped_inputs.append(inputs[i])

        # flip inputs and convert them back to their original type
        inputs = tf.concat([tf.cast(v, 'float32') for v in swapped_inputs], axis=-1)
        inputs = tf.map_fn(self._single_flip, [inputs, rand_flip], dtype=tf.float32)
        inputs = tf.split(inputs, self.list_n_channels, axis=-1)

        if self.several_inputs:
            return [tf.cast(v, t) for (t, v) in zip(types, inputs)]
        else:
            return tf.cast(inputs[0], types[0])

    def _single_swap(self, inputs):
        return K.switch(inputs[1], tf.gather(self.swap_lut, inputs[0]), inputs[0])

    @staticmethod
    def _single_flip(inputs):
        flip_axis = tf.where(inputs[1])
        return K.switch(tf.equal(tf.size(flip_axis), 0), inputs[0], tf.reverse(inputs[0], axis=flip_axis[..., 0]))


class SampleConditionalGMM(Layer):
    """This layer generates an image by sampling a Gaussian Mixture Model conditioned on a label map given as input.
    The parameters of the GMM are given as two additional inputs to the layer (means and standard deviations):
    image = SampleConditionalGMM(generation_labels)([label_map, means, stds])

    :param generation_labels: list of all possible label values contained in the input label maps.
    Must be a list or a 1D numpy array of size N, where N is the total number of possible label values.

    Layer inputs:
    label_map: input label map of shape [batchsize, shape_dim1, ..., shape_dimn, n_channel].
    All the values of label_map must be contained in generation_labels, but the input label_map doesn't necessarily have
    to contain all the values in generation_labels.
    means: tensor containing the mean values of all Gaussian distributions of the GMM.
           It must be of shape [batchsize, N, n_channel], and in the same order as generation label,
           i.e. the ith value of generation_labels will be associated to the ith value of means.
    stds: same as means but for the standard deviations of the GMM.
    """

    def __init__(self, generation_labels, **kwargs):
        self.generation_labels = generation_labels
        self.n_labels = None
        self.n_channels = None
        self.max_label = None
        self.indices = None
        self.shape = None
        super(SampleConditionalGMM, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["generation_labels"] = self.generation_labels
        return config

    def build(self, input_shape):

        # check n_labels and n_channels
        assert len(input_shape) == 3, 'should have three inputs: labels, means, std devs (in that order).'
        self.n_channels = input_shape[1][-1]
        self.n_labels = len(self.generation_labels)
        assert self.n_labels == input_shape[1][1], 'means should have the same number of values as generation_labels'
        assert self.n_labels == input_shape[2][1], 'stds should have the same number of values as generation_labels'

        # scatter parameters (to build mean/std lut)
        self.max_label = np.max(self.generation_labels) + 1
        indices = np.concatenate([self.generation_labels + self.max_label * i for i in range(self.n_channels)], axis=-1)
        self.shape = tf.convert_to_tensor([np.max(indices) + 1], dtype='int32')
        self.indices = tf.convert_to_tensor(utils.add_axis(indices, axis=[0, -1]), dtype='int32')

        self.built = True
        super(SampleConditionalGMM, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # reformat labels and scatter indices
        batch = tf.split(tf.shape(inputs[0]), [1, -1])[0]
        tmp_indices = tf.tile(self.indices, tf.concat([batch, tf.convert_to_tensor([1, 1], dtype='int32')], axis=0))
        labels = tf.concat([tf.cast(inputs[0], dtype='int32') + self.max_label * i for i in range(self.n_channels)], -1)

        # build mean map
        means = tf.concat([inputs[1][..., i] for i in range(self.n_channels)], 1)
        tile_shape = tf.concat([batch, tf.convert_to_tensor([1, ], dtype='int32')], axis=0)
        means = tf.tile(tf.expand_dims(tf.scatter_nd(tmp_indices, means, self.shape), 0), tile_shape)
        means_map = tf.map_fn(lambda x: tf.gather(x[0], x[1]), [means, labels], dtype=tf.float32)

        # same for stds
        stds = tf.concat([inputs[2][..., i] for i in range(self.n_channels)], 1)
        stds = tf.tile(tf.expand_dims(tf.scatter_nd(tmp_indices, stds, self.shape), 0), tile_shape)
        stds_map = tf.map_fn(lambda x: tf.gather(x[0], x[1]), [stds, labels], dtype=tf.float32)

        return stds_map * tf.random.normal(tf.shape(labels)) + means_map

    def compute_output_shape(self, input_shape):
        return input_shape[0] if (self.n_channels == 1) else tuple(list(input_shape[0][:-1]) + [self.n_channels])


class SampleResolution(Layer):
    """Build a random resolution tensor by sampling a uniform distribution of provided range.

    You can use this layer in the following ways:
        resolution = SampleConditionalGMM(min_resolution)() in this case resolution will be a tensor of shape (n_dims,),
        where n_dims is the length of the min_resolution parameter (provided as a list, see below).
        resolution = SampleConditionalGMM(min_resolution)(input), where input is a tensor for which the first dimension
        represents the batch_size. In this case resolution will be a tensor of shape (batchsize, n_dims,).

    :param min_resolution: list of length n_dims specifying the inferior bounds of the uniform distributions to
    sample from for each value.
    :param max_res_iso: If not None, all the values of resolution will be equal to the same value, which is randomly
    sampled at each minibatch in U(min_resolution, max_res_iso).
    :param max_res_aniso: If not None, we first randomly select a direction i in the range [0, n_dims-1], and we sample
    a value in the corresponding uniform distribution U(min_resolution[i], max_res_aniso[i]).
    The other values of resolution will be set to min_resolution.
    :param prob_iso: if both max_res_iso and max_res_aniso are specified, this allows to specify the probability of
    sampling an isotropic resolution (therefore using max_res_iso) with respect to anisotropic resolution
    (which would use max_res_aniso).
    :param prob_min: if not zero, this allows to return with the specified probability an output resolution equal
    to min_resolution.
    :param return_thickness: if set to True, this layer will also return a thickness value of the same shape as
    resolution, which will be sampled independently for each axis from the uniform distribution
    U(min_resolution, resolution).

    """

    def __init__(self,
                 min_resolution,
                 max_res_iso=None,
                 max_res_aniso=None,
                 prob_iso=0.1,
                 prob_min=0.05,
                 return_thickness=True,
                 **kwargs):

        self.min_res = min_resolution
        self.max_res_iso_input = max_res_iso
        self.max_res_iso = None
        self.max_res_aniso_input = max_res_aniso
        self.max_res_aniso = None
        self.prob_iso = prob_iso
        self.prob_min = prob_min
        self.return_thickness = return_thickness
        self.n_dims = len(self.min_res)
        self.add_batchsize = False
        self.min_res_tens = None
        super(SampleResolution, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["min_resolution"] = self.min_res
        config["max_res_iso"] = self.max_res_iso
        config["max_res_aniso"] = self.max_res_aniso
        config["prob_iso"] = self.prob_iso
        config["prob_min"] = self.prob_min
        config["return_thickness"] = self.return_thickness
        return config

    def build(self, input_shape):

        # check maximum resolutions
        assert ((self.max_res_iso_input is not None) | (self.max_res_aniso_input is not None)), \
            'at least one of maximum isotropic or anisotropic resolutions must be provided, received none'

        # reformat resolutions as numpy arrays
        self.min_res = np.array(self.min_res)
        if self.max_res_iso_input is not None:
            self.max_res_iso = np.array(self.max_res_iso_input)
            assert len(self.min_res) == len(self.max_res_iso), \
                'min and isotropic max resolution must have the same length, ' \
                'had {0} and {1}'.format(self.min_res, self.max_res_iso)
            if np.array_equal(self.min_res, self.max_res_iso):
                self.max_res_iso = None
        if self.max_res_aniso_input is not None:
            self.max_res_aniso = np.array(self.max_res_aniso_input)
            assert len(self.min_res) == len(self.max_res_aniso), \
                'min and anisotropic max resolution must have the same length, ' \
                'had {} and {}'.format(self.min_res, self.max_res_aniso)
            if np.array_equal(self.min_res, self.max_res_aniso):
                self.max_res_aniso = None

        # check prob iso
        if (self.max_res_iso is not None) & (self.max_res_aniso is not None) & (self.prob_iso == 0):
            raise Exception('prob iso is 0 while sampling either isotropic and anisotropic resolutions is enabled')

        if input_shape:
            self.add_batchsize = True

        self.min_res_tens = tf.convert_to_tensor(self.min_res, dtype='float32')

        self.built = True
        super(SampleResolution, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if not self.add_batchsize:
            shape = [self.n_dims]
            dim = tf.random.uniform(shape=(1, 1), minval=0, maxval=self.n_dims, dtype='int32')
            mask = tf.tensor_scatter_nd_update(tf.zeros([self.n_dims], dtype='bool'), dim,
                                               tf.convert_to_tensor([True], dtype='bool'))
        else:
            batch = tf.split(tf.shape(inputs), [1, -1])[0]
            tile_shape = tf.concat([batch, tf.convert_to_tensor([1], dtype='int32')], axis=0)
            self.min_res_tens = tf.tile(tf.expand_dims(self.min_res_tens, 0), tile_shape)

            shape = tf.concat([batch, tf.convert_to_tensor([self.n_dims], dtype='int32')], axis=0)
            indices = tf.stack([tf.range(0, batch[0]), tf.random.uniform(batch, 0, self.n_dims, dtype='int32')], 1)
            mask = tf.tensor_scatter_nd_update(tf.zeros(shape, dtype='bool'), indices, tf.ones(batch, dtype='bool'))

        # return min resolution as tensor if min=max
        if (self.max_res_iso is None) & (self.max_res_aniso is None):
            new_resolution = self.min_res_tens

        # sample isotropic resolution only
        elif (self.max_res_iso is not None) & (self.max_res_aniso is None):
            new_resolution_iso = tf.random.uniform(shape, minval=self.min_res, maxval=self.max_res_iso)
            new_resolution = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_min)),
                                      self.min_res_tens,
                                      new_resolution_iso)

        # sample anisotropic resolution only
        elif (self.max_res_iso is None) & (self.max_res_aniso is not None):
            new_resolution_aniso = tf.random.uniform(shape, minval=self.min_res, maxval=self.max_res_aniso)
            new_resolution = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_min)),
                                      self.min_res_tens,
                                      tf.where(mask, new_resolution_aniso, self.min_res_tens))

        # sample either anisotropic or isotropic resolution
        else:
            new_resolution_iso = tf.random.uniform(shape, minval=self.min_res, maxval=self.max_res_iso)
            new_resolution_aniso = tf.random.uniform(shape, minval=self.min_res, maxval=self.max_res_aniso)
            new_resolution = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_iso)),
                                      new_resolution_iso,
                                      tf.where(mask, new_resolution_aniso, self.min_res_tens))
            new_resolution = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_min)),
                                      self.min_res_tens,
                                      new_resolution)

        if self.return_thickness:
            return [new_resolution, tf.random.uniform(tf.shape(self.min_res_tens), self.min_res_tens, new_resolution)]
        else:
            return new_resolution

    def compute_output_shape(self, input_shape):
        if self.return_thickness:
            return [(None, self.n_dims)] * 2 if self.add_batchsize else [self.n_dims] * 2
        else:
            return (None, self.n_dims) if self.add_batchsize else self.n_dims


class GaussianBlur(Layer):
    """Applies gaussian blur to an input image.
    The input image is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param sigma: standard deviation of the blurring kernels to apply. Can be a number, a list of length n_dims, or
    a numpy array.
    :param random_blur_range: (optional) if not None, this introduces a randomness in the blurring kernels, where
    sigma is now multiplied by a coefficient dynamically sampled from a uniform distribution with bounds
    [1/random_blur_range, random_blur_range].
    :param use_mask: (optional) whether a mask of the input will be provided as an additional layer input. This is used
    to mask the blurred image, and to correct for edge blurring effects.

    example 1:
    output = GaussianBlur(sigma=0.5)(input) will isotropically blur the input with a gaussian kernel of std 0.5.

    example 2:
    if input is a tensor of shape [batchsize, 10, 100, 200, 2]
    output = GaussianBlur(sigma=[0.5, 1, 10])(input) will blur the input a different gaussian kernel in each dimension.

    example 3:
    output = GaussianBlur(sigma=0.5, random_blur_range=1.15)(input)
    will blur the input a different gaussian kernel in each dimension, as each dimension will be associated with
    a kernel, whose standard deviation will be uniformly sampled from [0.5/1.15; 0.5*1.15].

    example 4:
    output = GaussianBlur(sigma=0.5, use_mask=True)([input, mask])
    will 1) blur the input a different gaussian kernel in each dimension, 2) mask the blurred image with the provided
    mask, and 3) correct for edge blurring effects. If the provided mask is not of boolean type, it will be thresholded
    above positive values.
    """

    def __init__(self, sigma, random_blur_range=None, use_mask=False, **kwargs):
        self.sigma = utils.reformat_to_list(sigma)
        assert np.all(np.array(self.sigma) >= 0), 'sigma should be superior or equal to 0'
        self.use_mask = use_mask

        self.n_dims = None
        self.n_channels = None
        self.blur_range = random_blur_range
        self.stride = None
        self.separable = None
        self.kernels = None
        self.convnd = None
        super(GaussianBlur, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["random_blur_range"] = self.blur_range
        config["use_mask"] = self.use_mask
        return config

    def build(self, input_shape):

        # get shapes
        if self.use_mask:
            assert len(input_shape) == 2, 'please provide a mask as second layer input when use_mask=True'
            self.n_dims = len(input_shape[0]) - 2
            self.n_channels = input_shape[0][-1]
        else:
            self.n_dims = len(input_shape) - 2
            self.n_channels = input_shape[-1]

        # prepare blurring kernel
        self.stride = [1] * (self.n_dims + 2)
        self.sigma = utils.reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:  # fixed kernels
            self.kernels = l2i_et.gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

        self.built = True
        super(GaussianBlur, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None

        # redefine the kernels at each new step when blur_range is activated
        if self.blur_range is not None:
            self.kernels = l2i_et.gaussian_kernel(self.sigma, blur_range=self.blur_range, separable=self.separable)

        if self.separable:
            for k in self.kernels:
                if k is not None:
                    image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), k, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    if self.use_mask:
                        maskb = tf.cast(mask, 'float32')
                        maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), k, self.stride, 'SAME')
                                           for n in range(self.n_channels)], -1)
                        image = image / (maskb + K.epsilon())
                        image = tf.where(mask, image, tf.zeros_like(image))
        else:
            if any(self.sigma):
                image = tf.concat([self.convnd(tf.expand_dims(image[..., n], -1), self.kernels, self.stride, 'SAME')
                                   for n in range(self.n_channels)], -1)
                if self.use_mask:
                    maskb = tf.cast(mask, 'float32')
                    maskb = tf.concat([self.convnd(tf.expand_dims(maskb[..., n], -1), self.kernels, self.stride, 'SAME')
                                       for n in range(self.n_channels)], -1)
                    image = image / (maskb + K.epsilon())
                    image = tf.where(mask, image, tf.zeros_like(image))

        return image


class DynamicGaussianBlur(Layer):
    """Applies gaussian blur to an input image, where the standard deviation of the blurring kernel is provided as a
    layer input, which enables to perform dynamic blurring (i.e. the blurring kernel can vary at each minibatch).
    :param max_sigma: maximum value of the standard deviation that will be provided as input. This is used to compute
    the size of the blurring kernels. It must be provided as a list of length n_dims.
    :param random_blur_range: (optional) if not None, this introduces a randomness in the blurring kernels, where
    sigma is now multiplied by a coefficient dynamically sampled from a uniform distribution with bounds
    [1/random_blur_range, random_blur_range].

    example:
    blurred_image = DynamicGaussianBlur(max_sigma=[5.]*3, random_blurring_range=1.15)([image, sigma])
    will return a blurred version of image, where the standard deviation of each dimension (given as a tensor, and with
    values lower than 5 for each axis) is multiplied by a random coefficient uniformly sampled from [1/1.15; 1.15].
    """

    def __init__(self, max_sigma, random_blur_range=None, **kwargs):
        self.max_sigma = max_sigma
        self.n_dims = None
        self.n_channels = None
        self.convnd = None
        self.blur_range = random_blur_range
        self.separable = None
        super(DynamicGaussianBlur, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["max_sigma"] = self.max_sigma
        config["random_blur_range"] = self.blur_range
        return config

    def build(self, input_shape):
        assert len(input_shape) == 2, 'sigma should be provided as an input tensor for dynamic blurring'
        self.n_dims = len(input_shape[0]) - 2
        self.n_channels = input_shape[0][-1]
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)
        self.max_sigma = utils.reformat_to_list(self.max_sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.max_sigma)) > 5
        self.built = True
        super(DynamicGaussianBlur, self).build(input_shape)

    def call(self, inputs, **kwargs):
        image = inputs[0]
        sigma = inputs[-1]
        kernels = l2i_et.gaussian_kernel(sigma, self.max_sigma, self.blur_range, self.separable)
        if self.separable:
            for kernel in kernels:
                image = tf.map_fn(self._single_blur, [image, kernel], dtype=tf.float32)
        else:
            image = tf.map_fn(self._single_blur, [image, kernels], dtype=tf.float32)
        return image

    def _single_blur(self, inputs):
        if self.n_channels > 1:
            split_channels = tf.split(inputs[0], [1] * self.n_channels, axis=-1)
            blurred_channel = list()
            for channel in split_channels:
                blurred = self.convnd(tf.expand_dims(channel, 0), inputs[1], [1] * (self.n_dims + 2), padding='SAME')
                blurred_channel.append(tf.squeeze(blurred, axis=0))
            output = tf.concat(blurred_channel, -1)
        else:
            output = self.convnd(tf.expand_dims(inputs[0], 0), inputs[1], [1] * (self.n_dims + 2), padding='SAME')
            output = tf.squeeze(output, axis=0)
        return output


class MimicAcquisition(Layer):
    """
    Layer that takes an image as input, and simulates data that has been acquired at low resolution.
    The output is obtained by resampling the input twice:
     - first at a resolution given as an input (i.e. the "acquisition" resolution),
     - then at the output resolution (specified output shape).
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].

    :param volume_res: resolution of the provided inputs. Must be a 1-D numpy array with n_dims elements.
    :param min_subsample_res: lower bound of the acquisition resolutions to mimic (i.e. the input resolution must have
    values higher than min-subsample_res).
    :param resample_shape: shape of the output tensor
    :param build_dist_map: whether to return distance maps as outputs. These indicate the distance between each voxel
    and the nearest non-interpolated voxel (during the second resampling).
    :param prob_noise: probability to apply noise injection

    example 1:
    im_res = [1., 1., 1.]
    low_res = [1., 1., 3.]
    res = tf.convert_to_tensor([1., 1., 4.5])
    image is a tensor of shape (None, 256, 256, 256, 3)
    resample_shape = [256, 256, 256]
    output = MimicAcquisition(im_res, low_res, resample_shape)([image, res])
    output will be a tensor of shape (None, 256, 256, 256, 3), obtained by downsampling image to [1., 1., 4.5].
    and re-upsampling it at initial resolution (because resample_shape is equal to the input shape). In this example all
    examples of the batch will be downsampled to the same resolution (because res has no batch dimension).
    Note that the provided res must have higher values than min_low_res.

    example 2:
    im_res = [1., 1., 1.]
    min_low_res = [1., 1., 1.]
    res is a tensor of shape (None, 3), obtained for example by using the SampleResolution layer (see above).
    image is a tensor of shape (None, 256, 256, 256, 1)
    resample_shape = [128, 128, 128]
    output = MimicAcquisition(im_res, low_res, resample_shape)([image, res])
    output will be a tensor of shape (None, 128, 128, 128, 1), obtained by downsampling each examples of the batch to
    the matching resolution in res, and resampling them all to half the initial resolution.
    Note that the provided res must have higher values than min_low_res.
    """

    def __init__(self, volume_res, min_subsample_res, resample_shape, build_dist_map=False,
                 noise_std=0, prob_noise=0.95, **kwargs):

        # resolutions and dimensions
        self.volume_res = volume_res
        self.min_subsample_res = min_subsample_res
        self.n_dims = len(self.volume_res)
        self.n_channels = None
        self.add_batchsize = None

        # noise
        self.noise_std = noise_std
        self.prob_noise = prob_noise

        # input and output shapes
        self.inshape = None
        self.resample_shape = resample_shape

        # meshgrids for resampling
        self.down_grid = None
        self.up_grid = None

        # whether to return a map indicating the distance from the interpolated voxels, to acquired ones.
        self.build_dist_map = build_dist_map

        super(MimicAcquisition, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["volume_res"] = self.volume_res
        config["min_subsample_res"] = self.min_subsample_res
        config["resample_shape"] = self.resample_shape
        config["build_dist_map"] = self.build_dist_map
        config["noise_std"] = self.noise_std
        config["prob_noise"] = self.prob_noise
        return config

    def build(self, input_shape):

        # set up input shape and acquisition shape
        self.inshape = input_shape[0][1:]
        self.n_channels = input_shape[0][-1]
        self.add_batchsize = False if (input_shape[1][0] is None) else True
        down_tensor_shape = np.int32(np.array(self.inshape[:-1]) * self.volume_res / self.min_subsample_res)

        # build interpolation meshgrids
        self.down_grid = tf.expand_dims(tf.stack(nrn_utils.volshape_to_ndgrid(down_tensor_shape), -1), axis=0)
        self.up_grid = tf.expand_dims(tf.stack(nrn_utils.volshape_to_ndgrid(self.resample_shape), -1), axis=0)

        self.built = True
        super(MimicAcquisition, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # sort inputs
        assert len(inputs) == 2, 'inputs must have two items, the tensor to resample, and the downsampling resolution'
        vol = inputs[0]
        subsample_res = tf.cast(inputs[1], dtype='float32')
        vol = K.reshape(vol, [-1, *self.inshape])  # necessary for multi_gpu models
        batchsize = tf.split(tf.shape(vol), [1, -1])[0]
        tile_shape = tf.concat([batchsize, tf.ones([1], dtype='int32')], 0)

        # get downsampling and upsampling factors
        if self.add_batchsize:
            subsample_res = tf.tile(tf.expand_dims(subsample_res, 0), tile_shape)
        down_shape = tf.cast(tf.convert_to_tensor(np.array(self.inshape[:-1]) * self.volume_res, dtype='float32') /
                             subsample_res, dtype='int32')
        down_zoom_factor = tf.cast(down_shape / tf.convert_to_tensor(self.inshape[:-1]), dtype='float32')
        up_zoom_factor = tf.cast(tf.convert_to_tensor(self.resample_shape, dtype='int32') / down_shape, dtype='float32')

        # downsample
        down_loc = tf.tile(self.down_grid, tf.concat([batchsize, tf.ones([self.n_dims + 1], dtype='int32')], 0))
        down_loc = tf.cast(down_loc, 'float32') / l2i_et.expand_dims(down_zoom_factor, axis=[1] * self.n_dims)
        inshape_tens = tf.tile(tf.expand_dims(tf.convert_to_tensor(self.inshape[:-1]), 0), tile_shape)
        inshape_tens = l2i_et.expand_dims(inshape_tens, axis=[1] * self.n_dims)
        down_loc = K.clip(down_loc, 0., tf.cast(inshape_tens, 'float32'))
        vol = tf.map_fn(self._single_down_interpn, [vol, down_loc], tf.float32)

        # add noise with predefined probability
        if self.noise_std > 0:
            sample_shape = tf.concat([batchsize, tf.ones([self.n_dims], dtype='int32'),
                                      self.n_channels * tf.ones([1], dtype='int32')], 0)
            noise = tf.random.normal(tf.shape(vol), stddev=tf.random.uniform(sample_shape, maxval=self.noise_std))
            if self.prob_noise == 1:
                vol += noise
            else:
                vol = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_noise)), vol + noise, vol)

        # upsample
        up_loc = tf.tile(self.up_grid, tf.concat([batchsize, tf.ones([self.n_dims + 1], dtype='int32')], axis=0))
        up_loc = tf.cast(up_loc, 'float32') / l2i_et.expand_dims(up_zoom_factor, axis=[1] * self.n_dims)
        vol = tf.map_fn(self._single_up_interpn, [vol, up_loc], tf.float32)

        # return upsampled volume
        if not self.build_dist_map:
            return vol

        # return upsampled volumes with distance maps
        else:

            # get grid points
            floor = tf.math.floor(up_loc)
            ceil = tf.math.ceil(up_loc)

            # get distances of every voxel to higher and lower grid points for every dimension
            f_dist = up_loc - floor
            c_dist = ceil - up_loc

            # keep minimum 1d distances, and compute 3d distance to nearest grid point
            dist = tf.math.minimum(f_dist, c_dist) * l2i_et.expand_dims(subsample_res, axis=[1] * self.n_dims)
            dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(dist), axis=-1, keepdims=True))

            return [vol, dist]

    @staticmethod
    def _single_down_interpn(inputs):
        return nrn_utils.interpn(inputs[0], inputs[1], interp_method='nearest')

    @staticmethod
    def _single_up_interpn(inputs):
        return nrn_utils.interpn(inputs[0], inputs[1], interp_method='linear')

    def compute_output_shape(self, input_shape):
        output_shape = tuple([None] + self.resample_shape + [input_shape[0][-1]])
        return [output_shape] * 2 if self.build_dist_map else output_shape


class BiasFieldCorruption(Layer):
    """This layer applies a smooth random bias field to the input by applying the following steps:
    1) we first sample a value for the standard deviation of a centred normal distribution
    2) a small-size SVF is sampled from this normal distribution
    3) the small SVF is then resized with trilinear interpolation to image size
    4) it is rescaled to positive values by taking the voxel-wise exponential
    5) it is multiplied to the input tensor.
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].

    :param bias_field_std: maximum value of the standard deviation sampled in 1 (it will be sampled from the range
    [0, bias_field_std])
    :param bias_scale: ratio between the shape of the input tensor and the shape of the sampled SVF.
    :param same_bias_for_all_channels: whether to apply the same bias field to all the channels of the input tensor.
    :param prob: probability to apply this bias field corruption.
    """

    def __init__(self, bias_field_std=.5, bias_scale=.025, same_bias_for_all_channels=False, prob=0.95, **kwargs):

        # input shape
        self.several_inputs = False
        self.inshape = None
        self.n_dims = None
        self.n_channels = None

        # sampling shape
        self.std_shape = None
        self.small_bias_shape = None

        # bias field parameters
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.same_bias_for_all_channels = same_bias_for_all_channels
        self.prob = prob

        super(BiasFieldCorruption, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["bias_field_std"] = self.bias_field_std
        config["bias_scale"] = self.bias_scale
        config["same_bias_for_all_channels"] = self.same_bias_for_all_channels
        config["prob"] = self.prob
        return config

    def build(self, input_shape):

        # input shape
        if isinstance(input_shape, list):
            self.several_inputs = True
            self.inshape = input_shape
        else:
            self.inshape = [input_shape]
        self.n_dims = len(self.inshape[0]) - 2
        self.n_channels = self.inshape[0][-1]

        # sampling shapes
        self.std_shape = [1] * (self.n_dims + 1)
        self.small_bias_shape = utils.get_resample_shape(self.inshape[0][1:self.n_dims + 1], self.bias_scale, 1)
        if not self.same_bias_for_all_channels:
            self.std_shape[-1] = self.n_channels
            self.small_bias_shape[-1] = self.n_channels

        self.built = True
        super(BiasFieldCorruption, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if not self.several_inputs:
            inputs = [inputs]

        if self.bias_field_std > 0:

            # sampling shapes
            batchsize = tf.split(tf.shape(inputs[0]), [1, -1])[0]
            std_shape = tf.concat([batchsize, tf.convert_to_tensor(self.std_shape, dtype='int32')], 0)
            bias_shape = tf.concat([batchsize, tf.convert_to_tensor(self.small_bias_shape, dtype='int32')], axis=0)

            # sample small bias field
            bias_field = tf.random.normal(bias_shape, stddev=tf.random.uniform(std_shape, maxval=self.bias_field_std))

            # resize bias field and take exponential
            bias_field = nrn_layers.Resize(size=self.inshape[0][1:self.n_dims + 1], interp_method='linear')(bias_field)
            bias_field = tf.math.exp(bias_field)

            # apply bias field with predefined probability
            if self.prob == 1:
                return [tf.math.multiply(bias_field, v) for v in inputs]
            else:
                rand_trans = tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob))
                if self.several_inputs:
                    return [K.switch(rand_trans, tf.math.multiply(bias_field, v), v) for v in inputs]
                else:
                    return K.switch(rand_trans, tf.math.multiply(bias_field, inputs[0]), inputs[0])

        else:
            return inputs


class IntensityAugmentation(Layer):
    """This layer enables to augment the intensities of the input tensor, as well as to apply min_max normalisation.
    The following steps are applied (all are optional):
    1) white noise corruption, with a randomly sampled std dev.
    2) clip the input between two values
    3) min-max normalisation
    4) gamma augmentation (i.e. voxel-wise exponentiation by a randomly sampled power)
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].

    :param noise_std: maximum value of the standard deviation of the Gaussian white noise used in 1 (it will be sampled
    from the range [0, noise_std]). Set to 0 to skip this step.
    :param clip: clip the input tensor between the given values. Can either be: a number (in which case we clip between
    0 and the given value), or a list or a numpy array with two elements. Default is 0, where no clipping occurs.
    :param normalise: whether to apply min-max normalisation, to normalise between 0 and 1. Default is True.
    :param norm_perc: percentiles (between 0 and 1) of the sorted intensity values for robust normalisation. Can be:
    a number (in which case the robust minimum is the provided percentile of sorted values, and the maximum is the
    1 - norm_perc percentile), or a list/numpy array of 2 elements (percentiles for the minimum and maximum values).
    The minimum and maximum values are computed separately for each channel if separate_channels is True.
    Default is 0, where we simply take the minimum and maximum values.
    :param gamma_std: standard deviation of the normal distribution from which we sample gamma (in log domain).
    Default is 0, where no gamma augmentation occurs.
    :param contrast_inversion: whether to perform contrast inversion (i.e. 1 - x). If True, this is performed randomly
    for each element of the batch, as well as for each channel.
    :param separate_channels: whether to augment all channels separately. Default is True.
    :param prob_noise: probability to apply noise injection
    :param prob_gamma: probability to apply gamma augmentation
    """

    def __init__(self, noise_std=0, clip=0, normalise=True, norm_perc=0, gamma_std=0, contrast_inversion=False,
                 separate_channels=True, prob_noise=0.95, prob_gamma=1, **kwargs):

        # shape attributes
        self.n_dims = None
        self.n_channels = None
        self.flatten_shape = None
        self.expand_minmax_dim = None
        self.one = None

        # inputs
        self.noise_std = noise_std
        self.clip = clip
        self.clip_values = None
        self.normalise = normalise
        self.norm_perc = norm_perc
        self.perc = None
        self.gamma_std = gamma_std
        self.separate_channels = separate_channels
        self.contrast_inversion = contrast_inversion
        self.prob_noise = prob_noise
        self.prob_gamma = prob_gamma

        super(IntensityAugmentation, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["noise_std"] = self.noise_std
        config["clip"] = self.clip
        config["normalise"] = self.normalise
        config["norm_perc"] = self.norm_perc
        config["gamma_std"] = self.gamma_std
        config["separate_channels"] = self.separate_channels
        config["prob_noise"] = self.prob_noise
        config["prob_gamma"] = self.prob_gamma
        return config

    def build(self, input_shape):
        self.n_dims = len(input_shape) - 2
        self.n_channels = input_shape[-1]
        self.flatten_shape = np.prod(np.array(input_shape[1:-1]))
        self.flatten_shape = self.flatten_shape * self.n_channels if not self.separate_channels else self.flatten_shape
        self.expand_minmax_dim = self.n_dims if self.separate_channels else self.n_dims + 1
        self.one = tf.ones([1], dtype='int32')
        if self.clip:
            self.clip_values = utils.reformat_to_list(self.clip)
            self.clip_values = self.clip_values if len(self.clip_values) == 2 else [0, self.clip_values[0]]
        else:
            self.clip_values = None
        if self.norm_perc:
            self.perc = utils.reformat_to_list(self.norm_perc)
            self.perc = self.perc if len(self.perc) == 2 else [self.perc[0], 1 - self.perc[0]]
        else:
            self.perc = None

        self.built = True
        super(IntensityAugmentation, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # prepare shape for sampling the noise and gamma std dev (depending on whether we augment channels separately)
        batchsize = tf.split(tf.shape(inputs), [1, -1])[0]
        if (self.noise_std > 0) | (self.gamma_std > 0) | self.contrast_inversion:
            sample_shape = tf.concat([batchsize, tf.ones([self.n_dims], dtype='int32')], 0)
            if self.separate_channels:
                sample_shape = tf.concat([sample_shape, self.n_channels * self.one], 0)
            else:
                sample_shape = tf.concat([sample_shape, self.one], 0)
        else:
            sample_shape = None

        # add noise with predefined probability
        if self.noise_std > 0:
            noise_stddev = tf.random.uniform(sample_shape, maxval=self.noise_std)
            if self.separate_channels:
                noise = tf.random.normal(tf.shape(inputs), stddev=noise_stddev)
            else:
                noise = tf.random.normal(tf.shape(tf.split(inputs, [1, -1], -1)[0]), stddev=noise_stddev)
                noise = tf.tile(noise, tf.convert_to_tensor([1] * (self.n_dims + 1) + [self.n_channels]))
            if self.prob_noise == 1:
                inputs = inputs + noise
            else:
                inputs = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_noise)),
                                  inputs + noise, inputs)

        # clip images to given values
        if self.clip_values is not None:
            inputs = K.clip(inputs, self.clip_values[0], self.clip_values[1])

        # normalise
        if self.normalise:
            # define robust min and max by sorting values and taking percentile
            if self.perc is not None:
                if self.separate_channels:
                    shape = tf.concat([batchsize, self.flatten_shape * self.one, self.n_channels * self.one], 0)
                else:
                    shape = tf.concat([batchsize, self.flatten_shape * self.one], 0)
                intensities = tf.sort(tf.reshape(inputs, shape), axis=1)
                m = intensities[:, max(int(self.perc[0] * self.flatten_shape), 0), ...]
                M = intensities[:, min(int(self.perc[1] * self.flatten_shape), self.flatten_shape - 1), ...]
            # simple min and max
            else:
                m = K.min(inputs, axis=list(range(1, self.expand_minmax_dim + 1)))
                M = K.max(inputs, axis=list(range(1, self.expand_minmax_dim + 1)))
            # normalise
            m = l2i_et.expand_dims(m, axis=[1] * self.expand_minmax_dim)
            M = l2i_et.expand_dims(M, axis=[1] * self.expand_minmax_dim)
            inputs = tf.clip_by_value(inputs, m, M)
            inputs = (inputs - m) / (M - m + K.epsilon())

        # apply voxel-wise exponentiation with predefined probability
        if self.gamma_std > 0:
            gamma = tf.random.normal(sample_shape, stddev=self.gamma_std)
            if self.prob_gamma == 1:
                inputs = tf.math.pow(inputs, tf.math.exp(gamma))
            else:
                inputs = K.switch(tf.squeeze(K.less(tf.random.uniform([1], 0, 1), self.prob_gamma)),
                                  tf.math.pow(inputs, tf.math.exp(gamma)), inputs)

        # apply random contrast inversion
        if self.contrast_inversion:
            rand_invert = tf.less(tf.random.uniform(sample_shape, maxval=1), 0.5)
            split_channels = tf.split(inputs, [1] * self.n_channels, axis=-1)
            split_rand_invert = tf.split(rand_invert, [1] * self.n_channels, axis=-1)
            inverted_channel = list()
            for (channel, invert) in zip(split_channels, split_rand_invert):
                inverted_channel.append(tf.map_fn(self._single_invert, [channel, invert], dtype=channel.dtype))
            inputs = tf.concat(inverted_channel, -1)

        return inputs

    @staticmethod
    def _single_invert(inputs):
        return K.switch(tf.squeeze(inputs[1]), 1 - inputs[0], inputs[0])


class DiceLoss(Layer):
    """This layer computes the soft Dice loss between two tensors.
    These tensors are expected to have the same shape (one-hot encoding) [batch, size_dim1, ..., size_dimN, n_labels].
    The first input tensor is the GT and the second is the prediction: dice_loss = DiceLoss()([gt, pred])

    :param class_weights: (optional) if given, the loss is obtained by a weighted average of the Dice across labels.
    Must be a sequence or 1d numpy array of length n_labels. Can also be -1, where the weights are dynamically set to
    the inverse of the volume of each label in the ground truth.
    :param boundary_weights: (optional) bonus weight that we apply to the voxels close to boundaries between structures
    when computing the loss. Default is 0 where no boundary weighting is applied.
    :param boundary_dist: (optional) if boundary_weight is not 0, the extra boundary weighting is applied to all voxels
    within this distance to a region boundary. Default is 3.
    :param skip_background: (optional) whether to skip boundary weighting for the background class, as this may be
    redundant when we have several labels. This is only used if boundary_weight is not 0.
    :param enable_checks: (optional) whether to make sure that the 2 input tensors are probabilistic (i.e. the label
    probabilities sum to 1 at each voxel location). Default is True.
    """

    def __init__(self,
                 class_weights=None,
                 boundary_weights=0,
                 boundary_dist=3,
                 skip_background=True,
                 enable_checks=True,
                 **kwargs):

        self.class_weights = class_weights
        self.dynamic_weighting = False
        self.class_weights_tens = None
        self.boundary_weights = boundary_weights
        self.boundary_dist = boundary_dist
        self.skip_background = skip_background
        self.enable_checks = enable_checks
        self.spatial_axes = None
        self.avg_pooling_layer = None
        super(DiceLoss, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["class_weights"] = self.class_weights
        config["boundary_weights"] = self.boundary_weights
        config["boundary_dist"] = self.boundary_dist
        config["skip_background"] = self.skip_background
        config["enable_checks"] = self.enable_checks
        return config

    def build(self, input_shape):

        # get shape
        assert len(input_shape) == 2, 'DiceLoss expects 2 inputs to compute the Dice loss.'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        inshape = input_shape[0][1:]
        n_dims = len(inshape[:-1])
        n_labels = inshape[-1]
        self.spatial_axes = list(range(1, n_dims + 1))
        self.avg_pooling_layer = getattr(keras.layers, 'AvgPool%dD' % n_dims)
        self.skip_background = False if n_labels == 1 else self.skip_background

        # build tensor with class weights
        if self.class_weights is not None:
            if self.class_weights == -1:
                self.dynamic_weighting = True
            else:
                class_weights_tens = utils.reformat_to_list(self.class_weights, n_labels)
                class_weights_tens = tf.convert_to_tensor(class_weights_tens, 'float32')
                self.class_weights_tens = l2i_et.expand_dims(class_weights_tens, 0)

        self.built = True
        super(DiceLoss, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # make sure tensors are probabilistic
        gt = inputs[0]
        pred = inputs[1]
        if self.enable_checks:  # disabling is useful to, e.g., use incomplete label maps
            gt = K.clip(gt / (tf.math.reduce_sum(gt, axis=-1, keepdims=True) + tf.keras.backend.epsilon()), 0, 1)
            pred = K.clip(pred / (tf.math.reduce_sum(pred, axis=-1, keepdims=True) + tf.keras.backend.epsilon()), 0, 1)

        # compute dice loss for each label
        top = 2 * gt * pred
        bottom = tf.math.square(gt) + tf.math.square(pred)

        # apply boundary weighting (ie voxels close to region boundaries will be counted several times to compute Dice)
        if self.boundary_weights:
            avg = self.avg_pooling_layer(pool_size=2 * self.boundary_dist + 1, strides=1, padding='same')(gt)
            boundaries = tf.cast(avg > 0., 'float32') * tf.cast(avg < (1 / len(self.spatial_axes) - 1e-4), 'float32')
            if self.skip_background:
                boundaries_channels = tf.unstack(boundaries, axis=-1)
                boundaries = tf.stack([tf.zeros_like(boundaries_channels[0])] + boundaries_channels[1:], axis=-1)
            boundary_weights_tensor = 1 + self.boundary_weights * boundaries
            top *= boundary_weights_tensor
            bottom *= boundary_weights_tensor
        else:
            boundary_weights_tensor = None

        # compute loss
        top = tf.math.reduce_sum(top, self.spatial_axes)
        bottom = tf.math.reduce_sum(bottom, self.spatial_axes)
        dice = (top + tf.keras.backend.epsilon()) / (bottom + tf.keras.backend.epsilon())
        loss = 1 - dice

        # apply class weighting across labels. In this case loss will have shape (batch), otherwise (batch, n_labels).
        if self.dynamic_weighting:  # the weight of a class is the inverse of its volume in the gt
            if boundary_weights_tensor is not None:  # we account for the boundary weighting to compute volume
                self.class_weights_tens = 1 / tf.reduce_sum(gt * boundary_weights_tensor, self.spatial_axes)
            else:
                self.class_weights_tens = 1 / tf.reduce_sum(gt, self.spatial_axes)
        if self.class_weights_tens is not None:
            self. class_weights_tens /= tf.reduce_sum(self.class_weights_tens, -1)
            loss = tf.reduce_sum(loss * self.class_weights_tens, -1)

        return tf.math.reduce_mean(loss)

    def compute_output_shape(self, input_shape):
        return [[]]


class WeightedL2Loss(Layer):
    """This layer computes a L2 loss weighted by a specified factor (target_value) between two tensors.
    This is designed to be used on the layer before the softmax.
    The tensors are expected to have the same shape [batchsize, size_dim1, ..., size_dimN, n_labels].
    The first input tensor is the GT and the second is the prediction: wl2_loss = WeightedL2Loss()([gt, pred])

    :param target_value: target value for the layer before softmax: target_value when gt = 1, -target_value when gt = 0.
    """

    def __init__(self, target_value=5, **kwargs):
        self.target_value = target_value
        self.n_labels = None
        super(WeightedL2Loss, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["target_value"] = self.target_value
        return config

    def build(self, input_shape):
        assert len(input_shape) == 2, 'DiceLoss expects 2 inputs to compute the Dice loss.'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        self.n_labels = input_shape[0][-1]
        self.built = True
        super(WeightedL2Loss, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gt = inputs[0]
        pred = inputs[1]
        weights = tf.expand_dims(1 - gt[..., 0] + 1e-8, -1)
        return K.sum(weights * K.square(pred - self.target_value * (2 * gt - 1))) / (K.sum(weights) * self.n_labels)

    def compute_output_shape(self, input_shape):
        return [[]]


class CrossEntropyLoss(Layer):
    """This layer computes the cross-entropy loss between two tensors.
    These tensors are expected to have the same shape (one-hot encoding) [batch, size_dim1, ..., size_dimN, n_labels].
    The first input tensor is the GT and the second is the prediction: ce_loss = CrossEntropyLoss()([gt, pred])

    :param class_weights: (optional) if given, the loss is obtained by a weighted average of the Dice across labels.
    Must be a sequence or 1d numpy array of length n_labels. Can also be -1, where the weights are dynamically set to
    the inverse of the volume of each label in the ground truth.
    :param boundary_weights: (optional) bonus weight that we apply to the voxels close to boundaries between structures
    when computing the loss. Default is 0 where no boundary weighting is applied.
    :param boundary_dist: (optional) if boundary_weight is not 0, the extra boundary weighting is applied to all voxels
    within this distance to a region boundary. Default is 3.
    :param skip_background: (optional) whether to skip boundary weighting for the background class, as this may be
    redundant when we have several labels. This is only used if boundary_weight is not 0.
    :param enable_checks: (optional) whether to make sure that the 2 input tensors are probabilistic (i.e. the label
    probabilities sum to 1 at each voxel location). Default is True.
    """

    def __init__(self,
                 class_weights=None,
                 boundary_weights=0,
                 boundary_dist=3,
                 skip_background=True,
                 enable_checks=True,
                 **kwargs):

        self.class_weights = class_weights
        self.dynamic_weighting = False
        self.class_weights_tens = None
        self.boundary_weights = boundary_weights
        self.boundary_dist = boundary_dist
        self.skip_background = skip_background
        self.enable_checks = enable_checks
        self.spatial_axes = None
        self.avg_pooling_layer = None
        super(CrossEntropyLoss, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["class_weights"] = self.class_weights
        config["boundary_weights"] = self.boundary_weights
        config["boundary_dist"] = self.boundary_dist
        config["skip_background"] = self.skip_background
        config["enable_checks"] = self.enable_checks
        return config

    def build(self, input_shape):

        # get shape
        assert len(input_shape) == 2, 'CrossEntropy expects 2 inputs to compute the Dice loss.'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        inshape = input_shape[0][1:]
        n_dims = len(inshape[:-1])
        n_labels = inshape[-1]
        self.spatial_axes = list(range(1, n_dims + 1))
        self.avg_pooling_layer = getattr(keras.layers, 'AvgPool%dD' % n_dims)
        self.skip_background = False if n_labels == 1 else self.skip_background

        # build tensor with class weights
        if self.class_weights is not None:
            if self.class_weights == -1:
                self.dynamic_weighting = True
            else:
                class_weights_tens = utils.reformat_to_list(self.class_weights, n_labels)
                class_weights_tens = tf.convert_to_tensor(class_weights_tens, 'float32')
                self.class_weights_tens = l2i_et.expand_dims(class_weights_tens, [0] * (1 + n_dims))

        self.built = True
        super(CrossEntropyLoss, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # make sure tensors are probabilistic
        gt = inputs[0]
        pred = inputs[1]
        if self.enable_checks:  # disabling is useful to, e.g., use incomplete label maps
            gt = K.clip(gt / (tf.math.reduce_sum(gt, axis=-1, keepdims=True) + tf.keras.backend.epsilon()), 0, 1)
            pred = pred / (tf.math.reduce_sum(pred, axis=-1, keepdims=True) + tf.keras.backend.epsilon())
            pred = K.clip(pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())  # to avoid log(0)

        # compare prediction/target, ce has the same shape has the input tensors
        ce = -gt * tf.math.log(pred)

        # apply boundary weighting (ie voxels close to region boundaries will be counted several times to compute Dice)
        if self.boundary_weights:
            avg = self.avg_pooling_layer(pool_size=2 * self.boundary_dist + 1, strides=1, padding='same')(gt)
            boundaries = tf.cast(avg > 0., 'float32') * tf.cast(avg < (1 / len(self.spatial_axes) - 1e-4), 'float32')
            if self.skip_background:
                boundaries_channels = tf.unstack(boundaries, axis=-1)
                boundaries = tf.stack([tf.zeros_like(boundaries_channels[0])] + boundaries_channels[1:], axis=-1)
            boundary_weights_tensor = 1 + self.boundary_weights * boundaries
            ce *= boundary_weights_tensor
        else:
            boundary_weights_tensor = None

        # apply class weighting across labels. By the end of this, ce still has the same shape has the input tensors.
        if self.dynamic_weighting:  # the weight of a class is the inverse of its volume in the gt
            if boundary_weights_tensor is not None:  # we account for the boundary weighting to compute volume
                self.class_weights_tens = 1 / tf.reduce_sum(gt * boundary_weights_tensor, self.spatial_axes, True)
            else:
                self.class_weights_tens = 1 / tf.reduce_sum(gt, self.spatial_axes)
        if self.class_weights_tens is not None:
            self.class_weights_tens /= tf.reduce_sum(self.class_weights_tens, -1)
            ce = tf.reduce_sum(ce * self.class_weights_tens, -1)

        # sum along label axis, and take the mean along spatial dimensions
        ce = tf.math.reduce_mean(tf.math.reduce_sum(ce, axis=-1))

        return ce

    def compute_output_shape(self, input_shape):
        return [[]]


class MomentLoss(Layer):
    """This layer computes a moment loss between two tensors. Specifically, it computes the distance between the centres
    of gravity for all the channels of the two tensors, and then returns a value averaged across all channels.
    These tensors are expected to have the same shape [batch, size_dim1, ..., size_dimN, n_channels].
    The first input tensor is the GT and the second is the prediction: moment_loss = MomentLoss()([gt, pred])

    :param class_weights: (optional) if given, the loss is obtained by a weighted average of the Dice across labels.
    Must be a sequence or 1d numpy array of length n_labels. Can also be -1, where the weights are dynamically set to
    the inverse of the volume of each label in the ground truth.
    :param enable_checks: (optional) whether to make sure that the 2 input tensors are probabilistic (i.e. the label
    probabilities sum to 1 at each voxel location). Default is True.
    """

    def __init__(self, class_weights=None, enable_checks=False, **kwargs):
        self.class_weights = class_weights
        self.dynamic_weighting = False
        self.class_weights_tens = None
        self.enable_checks = enable_checks
        self.spatial_axes = None
        self.coordinates = None
        super(MomentLoss, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["class_weights"] = self.class_weights
        config["enable_checks"] = self.enable_checks
        return config

    def build(self, input_shape):

        # get shape
        assert len(input_shape) == 2, 'MomentLoss expects 2 inputs to compute the Dice loss.'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        inshape = input_shape[0][1:]
        n_dims = len(inshape[:-1])
        n_labels = inshape[-1]
        self.spatial_axes = list(range(1, n_dims + 1))

        # build coordinate meshgrid of size (1, dim1, dim2, ..., dimN, ndim, nchan)
        self.coordinates = tf.stack(nrn_utils.volshape_to_ndgrid(inshape[:-1]), -1)
        self.coordinates = tf.cast(l2i_et.expand_dims(tf.stack([self.coordinates] * n_labels, -1), 0), 'float32')

        # build tensor with class weights
        if self.class_weights is not None:
            if self.class_weights == -1:
                self.dynamic_weighting = True
            else:
                class_weights_tens = utils.reformat_to_list(self.class_weights, n_labels)
                class_weights_tens = tf.convert_to_tensor(class_weights_tens, 'float32')
                self.class_weights_tens = l2i_et.expand_dims(class_weights_tens, 0)

        self.built = True
        super(MomentLoss, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # make sure tensors are probabilistic
        gt = inputs[0]  # (B, dim1, dim2, ..., dimN, nchan)
        pred = inputs[1]
        if self.enable_checks:  # disabling is useful to, e.g., use incomplete label maps
            gt = gt / (tf.math.reduce_sum(gt, axis=-1, keepdims=True) + tf.keras.backend.epsilon())
            pred = pred / (tf.math.reduce_sum(pred, axis=-1, keepdims=True) + tf.keras.backend.epsilon())

        # compute loss
        gt_mean_coordinates = self._mean_coordinates(gt)  # (B, ndim, nchan)
        pred_mean_coordinates = self._mean_coordinates(pred)
        loss = tf.math.sqrt(tf.reduce_sum(tf.square(pred_mean_coordinates - gt_mean_coordinates), axis=1))  # (B, nchan)

        # apply class weighting across labels. In this case loss will have shape (batch), otherwise (batch, n_labels).
        if self.dynamic_weighting:  # the weight of a class is the inverse of its volume in the gt
            self.class_weights_tens = 1 / tf.reduce_sum(gt, self.spatial_axes)
        if self.class_weights_tens is not None:
            self.class_weights_tens /= tf.reduce_sum(self.class_weights_tens, -1)
            loss = tf.reduce_sum(loss * self.class_weights_tens, -1)

        return tf.math.reduce_mean(loss)

    def _mean_coordinates(self, tensor):
        tensor = l2i_et.expand_dims(tensor, axis=-2)  # (B, dim1, dim2, ..., dimN, 1, nchan)
        numerator = tf.reduce_sum(tensor * self.coordinates, axis=self.spatial_axes)  # (B, ndim, nchan)
        denominator = tf.reduce_sum(tensor, axis=self.spatial_axes) + tf.keras.backend.epsilon()
        return numerator / denominator

    def compute_output_shape(self, input_shape):
        return [[]]


class ResetValuesToZero(Layer):
    """This layer enables to reset given values to 0 within the input tensors.

    :param values: list of values to be reset to 0.

    example:
    input = tf.convert_to_tensor(np.array([[1, 0, 2, 2, 2, 2, 0],
                                           [1, 3, 3, 3, 3, 3, 3],
                                           [1, 0, 0, 0, 4, 4, 4]]))
    values = [1, 3]
    ResetValuesToZero(values)(input)
    >> [[0, 0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 4, 4]]
    """

    def __init__(self, values, **kwargs):
        assert values is not None, 'please provide correct list of values, received None'
        self.values = utils.reformat_to_list(values)
        self.values_tens = None
        self.n_values = len(values)
        super(ResetValuesToZero, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["values"] = self.values
        return config

    def build(self, input_shape):
        self.values_tens = tf.convert_to_tensor(self.values)
        self.built = True
        super(ResetValuesToZero, self).build(input_shape)

    def call(self, inputs, **kwargs):
        values = tf.cast(self.values_tens, dtype=inputs.dtype)
        for i in range(self.n_values):
            inputs = tf.where(tf.equal(inputs, values[i]), tf.zeros_like(inputs), inputs)
        return inputs


class ConvertLabels(Layer):
    """Convert all labels in a tensor by the corresponding given set of values.
    labels_converted = ConvertLabels(source_values, dest_values)(labels).
    labels must be an int32 tensor, and labels_converted will also be int32.

    :param source_values: list of all the possible values in labels. Must be a list or a 1D numpy array.
    :param dest_values: list of all the target label values. Must be ordered the same as source values:
    labels[labels == source_values[i]] = dest_values[i].
    If None (default), dest_values is equal to [0, ..., N-1], where N is the total number of values in source_values,
    which enables to remap label maps to [0, ..., N-1].
    """

    def __init__(self, source_values, dest_values=None, **kwargs):
        self.source_values = source_values
        self.dest_values = dest_values
        self.lut = None
        super(ConvertLabels, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["source_values"] = self.source_values
        config["dest_values"] = self.dest_values
        return config

    def build(self, input_shape):
        self.lut = tf.convert_to_tensor(utils.get_mapping_lut(self.source_values, dest=self.dest_values), dtype='int32')
        self.built = True
        super(ConvertLabels, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.gather(self.lut, tf.cast(inputs, dtype='int32'))


class PadAroundCentre(Layer):
    """Pad the input tensor to the specified shape with the given value.
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param pad_margin: margin to use for padding. The tensor will be padded by the provided margin on each side.
    Can either be a number (all axes padded with the same margin), or a  list/numpy array of length n_dims.
    example: if tensor is of shape [batch, x, y, z, n_channels] and margin=10, then the padded tensor will be of
    shape [batch, x+2*10, y+2*10, z+2*10, n_channels].
    :param pad_shape: shape to pad the tensor to. Can either be a number (all axes padded to the same shape), or a
    list/numpy array of length n_dims.
    :param value: value to pad the tensors with. Default is 0.
    """

    def __init__(self, pad_margin=None, pad_shape=None, value=0, **kwargs):
        self.pad_margin = pad_margin
        self.pad_shape = pad_shape
        self.value = value
        self.pad_margin_tens = None
        self.pad_shape_tens = None
        self.n_dims = None
        super(PadAroundCentre, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["pad_margin"] = self.pad_margin
        config["pad_shape"] = self.pad_shape
        config["value"] = self.value
        return config

    def build(self, input_shape):
        # input shape
        self.n_dims = len(input_shape) - 2
        shape = list(input_shape)
        shape[0] = 0
        shape[-1] = 0

        if self.pad_margin is not None:
            assert self.pad_shape is None, 'please do not provide a padding shape and margin at the same time.'

            # reformat padding margins
            pad = np.transpose(np.array([[0] + utils.reformat_to_list(self.pad_margin, self.n_dims) + [0]] * 2))
            self.pad_margin_tens = tf.convert_to_tensor(pad, dtype='int32')

        elif self.pad_shape is not None:
            assert self.pad_margin is None, 'please do not provide a padding shape and margin at the same time.'

            # pad shape
            tensor_shape = tf.cast(tf.convert_to_tensor(shape), 'int32')
            self.pad_shape_tens = np.array([0] + utils.reformat_to_list(self.pad_shape, length=self.n_dims) + [0])
            self.pad_shape_tens = tf.convert_to_tensor(self.pad_shape_tens, dtype='int32')
            self.pad_shape_tens = tf.math.maximum(tensor_shape, self.pad_shape_tens)

            # padding margin
            min_margins = (self.pad_shape_tens - tensor_shape) / 2
            max_margins = self.pad_shape_tens - tensor_shape - min_margins
            self.pad_margin_tens = tf.stack([min_margins, max_margins], axis=-1)

        else:
            raise Exception('please either provide a padding shape or a padding margin.')

        self.built = True
        super(PadAroundCentre, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.pad_margin_tens, mode='CONSTANT', constant_values=self.value)


class MaskEdges(Layer):
    """Reset the edges of a tensor to zero (i.e. with bands of zeros along the specified axes).
    The width of the zero-band is randomly drawn from a uniform distribution, whose range is given in boundaries.

    :param axes: axes along which to reset edges to zero. Can be an int (single axis), or a sequence.
    :param boundaries: numpy array of shape (len(axes), 4). Each row contains the two bounds of the uniform
    distributions from which we draw the width of the zero-bands on each side.
    Those bounds must be expressed in relative side (i.e. between 0 and 1).
    :return: a tensor of the same shape as the input, with bands of zeros along the specified axes.

    example:
    tensor=tf.constant([[[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]])  # shape = [1,10,10,1]
    axes=1
    boundaries = np.array([[0.2, 0.45, 0.85, 0.9]])

    In this case, we reset the edges along the 2nd dimension (i.e. the 1st dimension after the batch dimension),
    the 1st zero-band will expand from the 1st row to a number drawn from [0.2*tensor.shape[1], 0.45*tensor.shape[1]],
    and the 2nd zero-band will expand from a row drawn from [0.85*tensor.shape[1], 0.9*tensor.shape[1]], to the end of
    the tensor. A possible output could be:
    array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])  # shape = [1,10,10,1]
    """

    def __init__(self, axes, boundaries, prob_mask=1, **kwargs):
        self.axes = utils.reformat_to_list(axes, dtype='int')
        self.boundaries = utils.reformat_to_n_channels_array(boundaries, n_dims=4, n_channels=len(self.axes))
        self.prob_mask = prob_mask
        self.inputshape = None
        super(MaskEdges, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["axes"] = self.axes
        config["boundaries"] = self.boundaries
        config["prob_mask"] = self.prob_mask
        return config

    def build(self, input_shape):
        self.inputshape = input_shape
        self.built = True
        super(MaskEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # build mask
        mask = tf.ones_like(inputs)
        for i, axis in enumerate(self.axes):

            # select restricting indices
            axis_boundaries = self.boundaries[i, :]
            idx1 = tf.math.round(tf.random.uniform([1],
                                                   minval=axis_boundaries[0] * self.inputshape[axis],
                                                   maxval=axis_boundaries[1] * self.inputshape[axis]))
            idx2 = tf.math.round(tf.random.uniform([1],
                                                   minval=axis_boundaries[2] * self.inputshape[axis],
                                                   maxval=axis_boundaries[3] * self.inputshape[axis] - 1) - idx1)
            idx3 = self.inputshape[axis] - idx1 - idx2
            split_idx = tf.cast(tf.concat([idx1, idx2, idx3], axis=0), dtype='int32')

            # update mask
            split_list = tf.split(inputs, split_idx, axis=axis)
            tmp_mask = tf.concat([tf.zeros_like(split_list[0]),
                                  tf.ones_like(split_list[1]),
                                  tf.zeros_like(split_list[2])], axis=axis)
            mask = mask * tmp_mask

        # mask second_channel
        tensor = K.switch(tf.squeeze(K.greater(tf.random.uniform([1], 0, 1), 1 - self.prob_mask)),
                          inputs * mask,
                          inputs)

        return [tensor, mask]

    def compute_output_shape(self, input_shape):
        return [input_shape] * 2


class ImageGradients(Layer):

    def __init__(self, gradient_type='sobel', return_magnitude=False, **kwargs):

        self.gradient_type = gradient_type
        assert (self.gradient_type == 'sobel') | (self.gradient_type == '1-step_diff'), \
            'gradient_type should be either sobel or 1-step_diff, had %s' % self.gradient_type

        # shape
        self.n_dims = 0
        self.shape = None
        self.n_channels = 0

        # convolution params if sobel diff
        self.stride = None
        self.kernels = None
        self.convnd = None

        self.return_magnitude = return_magnitude

        super(ImageGradients, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["gradient_type"] = self.gradient_type
        config["return_magnitude"] = self.return_magnitude
        return config

    def build(self, input_shape):

        # get shapes
        self.n_dims = len(input_shape) - 2
        self.shape = input_shape[1:]
        self.n_channels = input_shape[-1]

        # prepare kernel if sobel gradients
        if self.gradient_type == 'sobel':
            self.kernels = l2i_et.sobel_kernels(self.n_dims)
            self.stride = [1] * (self.n_dims + 2)
            self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)
        else:
            self.kernels = self.convnd = self.stride = None

        self.built = True
        super(ImageGradients, self).build(input_shape)

    def call(self, inputs, **kwargs):

        image = inputs
        batchsize = tf.split(tf.shape(inputs), [1, -1])[0]
        gradients = list()

        # sobel method
        if self.gradient_type == 'sobel':
            # get sobel gradients in each direction
            for n in range(self.n_dims):
                gradient = image
                # apply 1D kernel in each direction (sobel kernels are separable), instead of applying a nD kernel
                for k in self.kernels[n]:
                    gradient = tf.concat([self.convnd(tf.expand_dims(gradient[..., n], -1), k, self.stride, 'SAME')
                                          for n in range(self.n_channels)], -1)
                gradients.append(gradient)

        # 1-step method, only supports 2 and 3D
        else:

            # get 1-step diff
            if self.n_dims == 2:
                gradients.append(image[:, 1:, :, :] - image[:, :-1, :, :])  # dx
                gradients.append(image[:, :, 1:, :] - image[:, :, :-1, :])  # dy

            elif self.n_dims == 3:
                gradients.append(image[:, 1:, :, :, :] - image[:, :-1, :, :, :])  # dx
                gradients.append(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])  # dy
                gradients.append(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])  # dz

            else:
                raise Exception('ImageGradients only support 2D or 3D tensors for 1-step diff, had: %dD' % self.n_dims)

            # pad with zeros to return tensors of the same shape as input
            for i in range(self.n_dims):
                tmp_shape = list(self.shape)
                tmp_shape[i] = 1
                zeros = tf.zeros(tf.concat([batchsize, tf.convert_to_tensor(tmp_shape, dtype='int32')], 0), image.dtype)
                gradients[i] = tf.concat([gradients[i], zeros], axis=i + 1)

        # compute total gradient magnitude if necessary, or concatenate different gradients along the channel axis
        if self.return_magnitude:
            gradients = tf.sqrt(tf.reduce_sum(tf.square(tf.stack(gradients, axis=-1)), axis=-1))
        else:
            gradients = tf.concat(gradients, axis=-1)

        return gradients

    def compute_output_shape(self, input_shape):
        if not self.return_magnitude:
            input_shape = list(input_shape)
            input_shape[-1] = self.n_dims
        return tuple(input_shape)


class RandomDilationErosion(Layer):
    """
    GPU implementation of binary dilation or erosion. The operation can be chosen to be always a dilation, or always an
    erosion, or randomly choosing between them for each element of the batch.
    The chosen operation is applied to the input with a given probability. Moreover, it is also possible to randomise
    the factor of the operation for each element of the mini-batch.
    :param min_factor: minimum possible value for the dilation/erosion factor. Must be an integer.
    :param max_factor: minimum possible value for the dilation/erosion factor. Must be an integer.
    Set it to the same value as min_factor to always perform dilation/erosion with the same factor.
    :param prob: probability with which to apply the selected operation to the input.
    :param operation: which operation to apply. Can be 'dilation' or 'erosion' or 'random'.
    :param return_mask: if operation is erosion and the input of this layer is a label map, we have the
    choice to either return the eroded label map or the mask (return_mask=True)
    """

    def __init__(self, min_factor, max_factor, max_factor_dilate=None, prob=1, operation='random', return_mask=False,
                 **kwargs):

        self.min_factor = min_factor
        self.max_factor = max_factor
        self.max_factor_dilate = max_factor_dilate if max_factor_dilate is not None else self.max_factor
        self.prob = prob
        self.operation = operation
        self.return_mask = return_mask
        self.n_dims = None
        self.inshape = None
        self.n_channels = None
        self.convnd = None
        super(RandomDilationErosion, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["min_factor"] = self.min_factor
        config["max_factor"] = self.max_factor
        config["max_factor_dilate"] = self.max_factor_dilate
        config["prob"] = self.prob
        config["operation"] = self.operation
        config["return_mask"] = self.return_mask
        return config

    def build(self, input_shape):

        # input shape
        self.inshape = input_shape
        self.n_dims = len(self.inshape) - 2
        self.n_channels = self.inshape[-1]

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

        self.built = True
        super(RandomDilationErosion, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # sample probability of applying operation. If random negative is erosion and positive is dilation
        batchsize = tf.split(tf.shape(inputs), [1, -1])[0]
        shape = tf.concat([batchsize, tf.convert_to_tensor([1], dtype='int32')], axis=0)
        if self.operation == 'dilation':
            prob = tf.random.uniform(shape, 0, 1)
        elif self.operation == 'erosion':
            prob = tf.random.uniform(shape, -1, 0)
        elif self.operation == 'random':
            prob = tf.random.uniform(shape, -1, 1)
        else:
            raise ValueError("operation should either be 'dilation' 'erosion' or 'random', had %s" % self.operation)

        # build kernel
        if self.min_factor == self.max_factor:
            dist_threshold = self.min_factor * tf.ones(shape, dtype='int32')
        else:
            if (self.max_factor == self.max_factor_dilate) | (self.operation != 'random'):
                dist_threshold = tf.random.uniform(shape, minval=self.min_factor, maxval=self.max_factor, dtype='int32')
            else:
                dist_threshold = tf.cast(tf.map_fn(self._sample_factor, [prob], dtype=tf.float32), dtype='int32')
        kernel = l2i_et.unit_kernel(dist_threshold, self.n_dims, max_dist_threshold=self.max_factor)

        # convolve input mask with kernel according to given probability
        mask = tf.cast(tf.cast(inputs, dtype='bool'), dtype='float32')
        mask = tf.map_fn(self._single_blur, [mask, kernel, prob], dtype=tf.float32)
        mask = tf.cast(mask, 'bool')

        if self.return_mask:
            return mask
        else:
            return inputs * tf.cast(mask, dtype=inputs.dtype)

    def _sample_factor(self, inputs):
        return tf.cast(K.switch(K.less(tf.squeeze(inputs[0]), 0),
                                tf.random.uniform((1,), self.min_factor, self.max_factor, dtype='int32'),
                                tf.random.uniform((1,), self.min_factor, self.max_factor_dilate, dtype='int32')),
                       dtype='float32')

    def _single_blur(self, inputs):
        # dilate...
        new_mask = K.switch(K.greater(tf.squeeze(inputs[2]), 1 - self.prob + 0.001),
                            tf.cast(tf.greater(tf.squeeze(self.convnd(tf.expand_dims(inputs[0], 0), inputs[1],
                                    [1] * (self.n_dims + 2), padding='SAME'), axis=0), 0.01), dtype='float32'),
                            inputs[0])
        # ...or erode
        new_mask = K.switch(K.less(tf.squeeze(inputs[2]), - (1 - self.prob + 0.001)),
                            1 - tf.cast(tf.greater(tf.squeeze(self.convnd(tf.expand_dims(1 - new_mask, 0), inputs[1],
                                        [1] * (self.n_dims + 2), padding='SAME'), axis=0), 0.01), dtype='float32'),
                            new_mask)
        return new_mask

    def compute_output_shape(self, input_shape):
        return input_shape
