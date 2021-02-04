"""This file regroups several custom keras layers used in the generation model:
    - RandomSpatialDeformation,
    - RandomCrop,
    - RandomFlip,
    - SampleConditionalGMM,
    - Sampleresolution,
    - GaussianBlur,
    - DynamicGaussianBlur,
    - MimicAcquisition,
    - BiasFieldCorruption,
    - IntensityAugmentation,
    - ResetValuesToZero,
    - PadAroundCentre,
    - MaskEdges
"""

# python imports
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

# project imports
from . import utils
from . import edit_tensors as l2i_et

# third-party imports
from ext.neuron import utils as nrn_utils
import ext.neuron.layers as nrn_layers


class RandomSpatialDeformation(Layer):

    """This layer spatially deforms one or several tensors with a combination of affine and elastic transformations.
    The input tensors are expected to have the same shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    The non linear deformation is obtained by:
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
    :param enable_90_rotations: (optional) wheter to rotate the input by a random angle chosen in {0, 90, 180, 270}.
    This is done regardless of the value of rotation_bounds. If true, a different value is sampled for each dimension.
    :param nonlin_std: (optional) maximum value of the standard deviation of the normal distribution from which we
    sample the small-size SVF. Set to 0 if you wish to completely turn the elastic deformation off.
    :param nonlin_shape_factor: (optional) if nonlin_std is not False, factor between the shapes of the input tensor
    and the shape of the input non-linear tensor.
    :param inter_method: (optional) interpolation method when deforming the input tensor. Can be 'linear', or 'nearest'
    """

    def __init__(self,
                 scaling_bounds=0.15,
                 rotation_bounds=10,
                 shearing_bounds=0.02,
                 translation_bounds=False,
                 enable_90_rotations=False,
                 nonlin_std=4.,
                 nonlin_shape_factor=.0625,
                 inter_method='linear',
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
        self.nonlin_shape_factor = nonlin_shape_factor

        # boolean attributes
        self.apply_affine_trans = None
        self.apply_elastic_trans = None

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
        config["nonlin_shape_factor"] = self.nonlin_shape_factor
        config["inter_method"] = self.inter_method
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            inputshape = [input_shape]
        else:
            self.n_inputs = len(input_shape)
            inputshape = input_shape
        self.inshape = inputshape[0][1:]
        self.n_dims = len(self.inshape) - 1
        self.built = True

        self.apply_affine_trans = (self.scaling_bounds is not False) | (self.rotation_bounds is not False) | \
                                  (self.shearing_bounds is not False) | (self.translation_bounds is not False) | \
                                  self.enable_90_rotations
        self.apply_elastic_trans = self.nonlin_std > 0
        assert (self.apply_affine_trans is not None) | self.apply_elastic_trans, \
            'affine_trans or elastic_trans should be provided'

        if self.apply_elastic_trans:
            self.small_shape = utils.get_resample_shape(self.inshape[:self.n_dims],
                                                        self.nonlin_shape_factor, self.n_dims)
        else:
            self.small_shape = None

        self.inter_method = utils.reformat_to_list(self.inter_method, length=self.n_inputs, dtype='str')

        super(RandomSpatialDeformation, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # reformat inputs and get its shape
        if self.n_inputs < 2:
            inputs = [inputs]
        types = [v.dtype for v in inputs]
        inputs = [tf.cast(v, dtype='float32') for v in inputs]
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]

        # initialise list of transfors to operate
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
        inputs = [nrn_layers.SpatialTransformer(m)([v] + list_trans) for (m, v) in zip(self.inter_method, inputs)]
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

    """This function flips the input tensors along the specified axes with a probability of 0.5.
    The input tensors are expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    If specified, this layer can also swap all values, such that the flip tensors stay consistent with spatial the
    native spatial orientation (especially convenient when flipping label maps in the righ/left dimension).
    :param flip_axis: integer, or list of integers specifying the dimensions along which to flip. The values exclude the
    batch dimension (e.g. 0 will flip the tensor along the first axis after the batch dimension). Default is None, where
    the tensors can be flipped along any of the axes (except batch and channel axes).
    :param swap_labels: list of booleans to specify wether to swap the values of each input
    :param label_list: if swap_labels is True, list of all labels contained in labels. Must be ordered as follows, first
     the neutral labels (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: if swap_labels is True, number of non-sided labels

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

    example 4:
    if labels is the same as in the input of example 3, and image is a float32 image, then we can swap consistently both
    the labels and the image with:
    labels, image = RandomFlip(flip_axis=1, swap_labels=[True, False], label_list=label_list,
                               n_neutral_labels=n_neutral_labels)([labels, image]])
    """

    def __init__(self, flip_axis=None, swap_labels=False, label_list=None, n_neutral_labels=None, **kwargs):

        # shape attributes
        self.several_inputs = True
        self.n_dims = None
        self.list_n_channels = None

        # axis along which to flip
        self.flip_axis = utils.reformat_to_list(flip_axis)

        # wether to swap labels, and corresponding label list
        self.swap_labels = utils.reformat_to_list(swap_labels)
        self.label_list = label_list
        self.n_neutral_labels = n_neutral_labels
        self.swapped_label_list = None

        super(RandomFlip, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["flip_axis"] = self.flip_axis
        config["swap_labels"] = self.swap_labels
        config["label_list"] = self.label_list
        config["n_neutral_labels"] = self.n_neutral_labels
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
                self.swapped_label_list = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))

        self.built = True
        super(RandomFlip, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # convert inputs to list, and get each input type
        if not self.several_inputs:
            inputs = [inputs]
        types = [v.dtype for v in inputs]

        # sample boolean for each element of the batch
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]
        rand_flip = K.greater(tf.random.uniform(tf.concat([batchsize, tf.ones(1, dtype='int32')], axis=0), 0, 1), 0.5)

        # swap r/l labels if necessary
        swapped_inputs = list()
        for i in range(len(inputs)):
            if self.swap_labels[i]:
                swapped_inputs.append(tf.map_fn(self._single_swap, [inputs[i], rand_flip], dtype=tf.int32))
            else:
                swapped_inputs.append(inputs[i])

        # flip inputs and convert them back to their original type
        inputs = tf.concat([tf.cast(v, 'float32') for v in swapped_inputs], axis=-1)
        inputs = tf.map_fn(self._single_flip, [inputs, rand_flip], dtype=tf.float32)
        inputs = tf.split(inputs, self.list_n_channels, axis=-1)

        return [tf.cast(v, t) for (t, v) in zip(types, inputs)]

    def _single_swap(self, inputs):
        return K.switch(inputs[1],
                        tf.gather(tf.convert_to_tensor(self.swapped_label_list, dtype='int32'), inputs[0]),
                        inputs[0])

    def _single_flip(self, inputs):
        if self.flip_axis is None:
            flip_axis = tf.random.uniform([1], 0, self.n_dims, dtype='int32')
        else:
            idx = tf.squeeze(tf.random.uniform([1], 0, len(self.flip_axis), dtype='int32'))
            flip_axis = tf.expand_dims(tf.convert_to_tensor(self.flip_axis, dtype='int32')[idx], axis=0)
        return K.switch(inputs[1], K.reverse(inputs[0], axes=flip_axis), inputs[0])


class SampleConditionalGMM(Layer):
    """This layer generates an image by sampling a Gaussian Mixture Model conditioned on a label map given as input.
    The parameters of the GMM are given as two additional inputs to the layer (means and standard deviations):
    image = SampleConditionalGMM()([label_map, means, stds])

    label_map: input label map of shape [batchsize, shape_dim1, ..., shape_dimn, channel]. Its values must be in
               [0, ..., N-1], where N is the number of label values in label_map.
    means: tensor containing the mean values of all Gaussian distributions of the GMM.
           It must be of shape [batchsize, n_gaussians, channel]. All the label values must have a corresponding
           Gaussian, but not all gaussians must be present in label_map (i.e. n_gaussians>=N).
    stds: same as means but for the standard deviations of the GMM.
    """

    def __init__(self, **kwargs):
        self.n_labels = None
        self.n_channels = None
        super(SampleConditionalGMM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, 'should have three inputs: labels, means, std devs (in that order).'
        self.n_labels = input_shape[1][1]
        self.n_channels = input_shape[1][-1]
        self.built = True
        super(SampleConditionalGMM, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # reformat inputs
        labels = tf.concat([tf.cast(inputs[0], dtype='int32') + self.n_labels * i for i in range(self.n_channels)], -1)
        means = tf.concat([inputs[1][..., i] for i in range(self.n_channels)], 1)
        stds = tf.concat([inputs[2][..., i] for i in range(self.n_channels)], 1)

        # build mean and std maps
        means_map = tf.map_fn(lambda x: tf.gather(x[1], x[0]), [labels, means], dtype=tf.float32)
        stds_map = tf.map_fn(lambda x: tf.gather(x[1], x[0]), [labels, stds], dtype=tf.float32)

        return stds_map * tf.random.normal(tf.shape(labels)) + means_map

    def compute_output_shape(self, input_shape):
        return input_shape[0] if (self.n_channels == 1) else tuple(list(input_shape[0][:-1]) + [self.n_channels])


class SampleResolution(Layer):
    """Pad the input tensor to the specified shape with the given value.
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param pad_shape: shape to pad the tensor to. Can either be a number (all axes padded to the same shape), or a
    list/numpy array of length n_dims.
    :param value: value to pad the tensors with. Default is 0.
    """

    def __init__(self, min_resolution, max_resolution, prob_min=0, return_thickness=False, **kwargs):
        self.min_res = min_resolution
        self.max_res = max_resolution
        self.prob_min = prob_min
        self.return_thickness = return_thickness
        self.n_dims = len(self.min_res)
        self.add_batchsize = False
        self.min_res_tens = None
        super(SampleResolution, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["min_resolution"] = self.min_res
        config["max_resolution"] = self.max_res
        config["prob_min"] = self.prob_min
        config["return_thickness"] = self.return_thickness
        return config

    def build(self, input_shape):

        # check dimension
        assert len(self.min_res) == len(self.max_res), \
            'min and max resolution must have the same length, had {0} and {1}'.format(self.min_res, self.max_res)

        # make sure min and max resolutions are np array
        self.min_res = np.array(self.min_res)
        self.max_res = np.array(self.max_res)

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
        if np.array_equal(self.min_res, self.max_res):
            new_resolution = self.min_res_tens
        else:
            # sample new resolution for each dimension
            new_resolution = tf.random.uniform(shape, minval=self.min_res, maxval=self.max_res)
            new_resolution = K.switch(tf.squeeze(K.greater(tf.random.uniform([1], 0, 1), 1 - self.prob_min)),
                                      self.min_res_tens,
                                      tf.where(mask, new_resolution, self.min_res_tens))

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
    :param use_mask: (optional) whether a mask of the input will be provided as an additionnal layer input. This is used
    to mask the blurred image, and to correct for edge blurring effects.

    example 1:
    output = GaussianBlur(sigma=0.5)(input) will isotropically blur the input with a gaussian kernel of std 0.5.

    example 2:
    if input is a tensor of shape [batchsize, 10, 100, 200, 2]
    output = GaussianBlur(sigma=[0.5, 1, 10])(input) will blur the input a different gaussian kernel in each dimension.

    example 3:
    output = GaussianBlur(sigma=0.5, random_blur_range=1.15)(input)
    will blur the input a different gaussian kernel in each dimension, as each dimension will be associated with a
    a kernel, whose standard deviation will be uniformly sampled from [0.5/1.15; 0.5*1.15].

    example 4:
    output = GaussianBlur(sigma=0.5, use_mask=True)([input, mask])
    will 1) blur the input a different gaussian kernel in each dimension, 2) mask the blurred image with the provided
    mask, and 3) correct for edge blurring effects. If the provided mask is not of boolean type, it will thresholded
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
        super().__init__(**kwargs)

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
        self.stride = [1]*(self.n_dims+2)
        self.sigma = utils.reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:  # fixed kernels
            self.kernels = l2i_et.gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

    def call(self, inputs, **kwargs):

        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None

        # redefine the kernels at each new step when blur_range is activated
        if self.kernels is None:
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
    will return a blurred version of image, where the standard deviation of each dimension (given as an tensor, and with
    values lower than 5 for each axis) is multiplied by a random coefficient uniformly sampled from [1/1.15; 1.15].
    """

    def __init__(self, max_sigma, random_blur_range=None, **kwargs):
        self.max_sigma = max_sigma
        self.n_dims = None
        self.n_channels = None
        self.convnd = None
        self.blur_range = random_blur_range
        self.separable = None
        super().__init__(**kwargs)

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
        blurred_channel = list()
        for n in range(self.n_channels):
            blurred = self.convnd(tf.expand_dims(inputs[0], 0), inputs[1], [1] * (self.n_dims + 2), padding='SAME')
            blurred_channel.append(tf.squeeze(blurred, axis=0))
        return tf.concat(blurred_channel, -1)


class MimicAcquisition(Layer):
    """
    Layer that takes an image as input, and simulates data that has been acquired at low resolution.
    The output is obtained by resampling the input twice:
     - first at a resolution given as an input (i.e. the "acquisition" resolution),
     - then at the output resolution (specified output shape).
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].

    :param volume_res: resolution of the provided inputs. Must be a 1-D numpy array with n_dims elements.
    :param min_subsample_res: lower bound of the acquisition resolutions to mimic (i.e. the input resolution must have
    values higher than min-subseample_res).
    :param resample_shape: shape of the output tensor
    :param build_dist_map: whether to return distance maps as outputs. These indicate the distance between each voxel
    and the nearest non-interpolated voxel (during the second resampling).

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
    the matching resolution in res, and resanpling them all to half the initial resolution.
    Note that the provided res must have higher values than min_low_res.
    """

    def __init__(self, volume_res, min_subsample_res, resample_shape, build_dist_map=False, **kwargs):

        # resolutions and dimensions
        self.volume_res = volume_res
        self.min_subsample_res = min_subsample_res
        self.ndims = len(self.volume_res)
        self.add_batchsize = None

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
        return config

    def build(self, input_shape):

        # set up input shape and acquisistion shape
        self.inshape = input_shape[0][1:]
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
        down_loc = tf.tile(self.down_grid, tf.concat([batchsize, tf.ones([self.ndims + 1], dtype='int32')], 0))
        down_loc = tf.cast(down_loc, 'float32') / l2i_et.expand_dims(down_zoom_factor, axis=[1] * self.ndims)
        inshape_tens = tf.tile(tf.expand_dims(tf.convert_to_tensor(self.inshape[:-1]), 0), tile_shape)
        inshape_tens = l2i_et.expand_dims(inshape_tens, axis=[1] * self.ndims)
        down_loc = K.clip(down_loc, 0., tf.cast(inshape_tens, 'float32'))
        vol = tf.map_fn(self._single_down_interpn, [vol, down_loc], tf.float32)

        # upsample
        up_loc = tf.tile(self.up_grid, tf.concat([batchsize, tf.ones([self.ndims + 1], dtype='int32')], axis=0))
        up_loc = tf.cast(up_loc, 'float32') / l2i_et.expand_dims(up_zoom_factor, axis=[1] * self.ndims)
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
            dist = tf.math.minimum(f_dist, c_dist) * l2i_et.expand_dims(subsample_res, axis=[1] * self.ndims)
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
    4) it is rescaled to postive values by taking the voxel-wise exponential
    5) it is multiplied to the input tensor.
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].

    :param bias_field_std: maximum value of the standard deviation sampled in 1 (it will be sampled from the range
    [0, bias_field_std])
    :param bias_shape_factor: ratio between the shape of the input tensor and the shape of the sampled SVF.
    :param same_bias_for_all_channels: whether to apply the same bias field to all the channels of the input tensor.
    """

    def __init__(self, bias_field_std=.5, bias_shape_factor=.025, same_bias_for_all_channels=False, **kwargs):

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
        self.bias_shape_factor = bias_shape_factor
        self.same_bias_for_all_channels = same_bias_for_all_channels

        super(BiasFieldCorruption, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["bias_field_std"] = self.bias_field_std
        config["bias_shape_factor"] = self.bias_shape_factor
        config["same_bias_for_all_channels"] = self.same_bias_for_all_channels
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
        self.small_bias_shape = utils.get_resample_shape(self.inshape[0][1:self.n_dims + 1], self.bias_shape_factor, 1)
        if not self.same_bias_for_all_channels:
            self.std_shape[-1] = self.n_channels
            self.small_bias_shape[-1] = self.n_channels

        self.built = True
        super(BiasFieldCorruption, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if not self.several_inputs:
            inputs = [inputs]

        # sampling shapes
        batchsize = tf.split(tf.shape(inputs[0]), [1, -1])[0]
        std_shape = tf.concat([batchsize, tf.convert_to_tensor(self.std_shape, dtype='int32')], 0)
        bias_shape = tf.concat([batchsize, tf.convert_to_tensor(self.small_bias_shape, dtype='int32')], axis=0)

        # sample small bias field
        bias_field = tf.random.normal(bias_shape, stddev=tf.random.uniform(std_shape, maxval=self.bias_field_std))

        # resize bias field and take exponential
        bias_field = nrn_layers.Resize(size=self.inshape[0][1:self.n_dims + 1], interp_method='linear')(bias_field)
        bias_field = tf.math.exp(bias_field)

        return [tf.math.multiply(bias_field, v) for v in inputs]


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
    :param clip: If not False, this enables to clip the input tensor between the given values. This can either be:
    a number (in which case we clip between 0 and the given value), or a list or a numpy array with two elements.
    :param normalise: whether to apply min-max normalistion. Default is True.
    :param gamma_std: standard deviation of the normal distribution from which we sample gamma (in log domain).
    Set to 0 to skip this step.
    :param separate_channels: whether to augment all channels separately. Default is True.
    """

    def __init__(self, noise_std=0, clip=None, normalise=True, norm_perc=0, gamma_std=.4, separate_channels=True,
                 **kwargs):

        # shape attributes
        self.n_dims = None
        self.n_channels = None
        self.flatten_shape = None
        self.expand_minmax_dim = None
        self.one = None

        # inputs
        self.noise_std = noise_std
        self.clip = clip
        self.normalise = normalise
        self.norm_perc = norm_perc
        self.gamma_std = gamma_std
        self.separate_channels = separate_channels

        super(IntensityAugmentation, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["noise_std"] = self.noise_std
        config["clip"] = self.clip
        config["normalise"] = self.normalise
        config["norm_perc"] = self.norm_perc
        config["gamma_std"] = self.gamma_std
        config["separate_channels"] = self.separate_channels
        return config

    def build(self, input_shape):
        self.n_dims = len(input_shape) - 2
        self.n_channels = input_shape[-1]
        self.flatten_shape = np.prod(np.array(input_shape[1:-1]))
        self.flatten_shape = self.flatten_shape * self.n_channels if not self.separate_channels else self.flatten_shape
        self.expand_minmax_dim = self.n_dims if self.separate_channels else self.n_dims + 1
        self.one = tf.ones([1], dtype='int32')
        if self.clip is None:
            self.clip = [0, 300]
        elif self.clip is not False:
            self.clip = utils.reformat_to_list(self.clip)
            self.clip = self.clip if len(self.clip) == 2 else [0, self.clip[0]]
        super(IntensityAugmentation, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # prepare shape for sampling the noise and gamma std dev (depending on whether we augment channels separately)
        batchsize = tf.split(tf.shape(inputs), [1, -1])[0]
        if (self.noise_std > 0) | (self.gamma_std > 0):
            sample_shape = tf.concat([batchsize, tf.ones([self.n_dims], dtype='int32')], 0)
            if self.separate_channels:
                sample_shape = tf.concat([sample_shape, self.n_channels * self.one], 0)
            else:
                sample_shape = tf.concat([sample_shape, self.one], 0)
        else:
            sample_shape = None

        # add noise
        if self.noise_std > 0:
            noise_stddev = tf.random.uniform(sample_shape, maxval=self.noise_std)
            if self.separate_channels:
                noise = tf.random.normal(tf.shape(inputs), stddev=noise_stddev)
            else:
                noise = tf.random.normal(tf.shape(tf.split(inputs, [1, -1], -1)[0]), stddev=noise_stddev)
                noise = tf.tile(noise, tf.convert_to_tensor([1] * (self.n_dims + 1) + [self.n_channels]))
            inputs = inputs + noise

        # clip images to given values
        if self.clip is not False:
            inputs = K.clip(inputs, self.clip[0], self.clip[1])

        # normalise
        if self.normalise:
            # define robust min and max by sorting values and taking percentile
            if self.norm_perc > 0:
                if self.separate_channels:
                    shape = tf.concat([batchsize, self.flatten_shape * self.one, self.n_channels * self.one], 0)
                else:
                    shape = tf.concat([batchsize, self.flatten_shape * self.one], 0)
                intensities = tf.sort(tf.reshape(inputs, shape), axis=1)
                m = intensities[:, max(int(self.norm_perc * self.flatten_shape), 0), ...]
                M = intensities[:, min(int((1 - self.norm_perc) * self.flatten_shape), self.flatten_shape - 1), ...]
            # simple min and max
            else:
                m = K.min(inputs, axis=list(range(1, self.expand_minmax_dim + 1)))
                M = K.max(inputs, axis=list(range(1, self.expand_minmax_dim + 1)))
            # normalise
            m = l2i_et.expand_dims(m, axis=[1] * self.expand_minmax_dim)
            M = l2i_et.expand_dims(M, axis=[1] * self.expand_minmax_dim)
            inputs = (inputs - m) / (M - m)

        # apply voxel-wise exponentiation
        if self.gamma_std > 0:
            inputs = tf.math.pow(inputs, tf.math.exp(tf.random.normal(sample_shape, stddev=self.gamma_std)))

        return inputs


class DiceLoss(Layer):
    """This layer computes the Dice loss between two tensors. These tensors are expected to 1) have the same shape, and
    2) be probabilistic, i.e. they must have the same shape [batchsize, size_dim1, ..., size_dimN, n_labels] where
    n_labels is the number of labels for which we compute the Dice loss."""

    def __init__(self, **kwargs):
        self.inshape = None
        super(DiceLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2, 'DiceLoss expects 2 inputs to compute the Dice loss.'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        self.inshape = input_shape[0][1:]
        self.built = True
        super(DiceLoss, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # make sure tensors are probabilistic
        x = inputs[0]
        y = inputs[1]
        x = K.clip(x / tf.math.reduce_sum(x, axis=-1, keepdims=True), 0, 1)
        y = K.clip(y / tf.math.reduce_sum(y, axis=-1, keepdims=True), 0, 1)

        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * x * y, axis=list(range(1, len(self.inshape))))
        bottom = tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, len(self.inshape))))
        last_tensor = top / bottom

        return K.mean(1 - last_tensor)

    def compute_output_shape(self, input_shape):
        return [[]]


class WeightedL2Loss(Layer):
    """This layer computes a L2 loss weighted by a specified factor between two tensors.
    These tensors are expected to have the same shape [batchsize, size_dim1, ..., size_dimN, n_labels]
    where n_labels is the number of labels for which we compute the loss.
    Importantly, the first input tensor is the GT, whereas the second is the prediction."""

    def __init__(self, target_value, background_weight=1e-4, **kwargs):
        self.target_value = target_value
        self.background_weight = background_weight
        self.n_labels = None
        super(WeightedL2Loss, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["target_value"] = self.target_value
        config["background_weight"] = self.background_weight
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
        weights = tf.expand_dims(1 - gt[..., 0] + self.background_weight, -1)
        return K.sum(weights * K.square(pred - self.target_value * (2 * gt - 1))) / (K.sum(weights) * self.n_labels)

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


class PadAroundCentre(Layer):
    """Pad the input tensor to the specified shape with the given value.
    The input tensor is expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param pad_shape: shape to pad the tensor to. Can either be a number (all axes padded to the same shape), or a
    list/numpy array of length n_dims.
    :param value: value to pad the tensors with. Default is 0.
    """

    def __init__(self, pad_shape, value=0, **kwargs):
        self.pad_shape = pad_shape
        self.pad_shape_tens = None
        self.value = value
        self.n_dims = None
        self.pad_margins = None
        super(PadAroundCentre, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["pad_shape"] = self.pad_shape
        config["value"] = self.value
        return config

    def build(self, input_shape):
        # input shape
        self.n_dims = len(input_shape) - 2
        input_shape[0] = 0
        input_shape[0 - 1] = 0
        tensor_shape = tf.cast(tf.convert_to_tensor(input_shape), 'int32')

        # pad shape
        self.pad_shape_tens = np.array([0] + utils.reformat_to_list(self.pad_shape, length=self.n_dims) + [0])
        self.pad_shape_tens = tf.cast(tf.convert_to_tensor(self.pad_shape_tens), 'int32')
        self.pad_shape_tens = tf.math.maximum(tensor_shape, self.pad_shape_tens)

        # padding margin
        min_margins = (self.pad_shape_tens - tensor_shape) / 2
        max_margins = self.pad_shape_tens - tensor_shape - min_margins
        self.pad_margins = tf.stack([min_margins, max_margins], axis=-1)

        self.built = True
        super(PadAroundCentre, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.pad_margins, mode='CONSTANT', constant_values=self.value)


class MaskEdges(Layer):
    """Reset the edges of a tensor to zero (i.e. with bands of zeros along the specified axes).
    The width of the zero-band is randomly drawn from a uniform distribution, whose range is given in boundaries.

    :param axes: axes along which to reset edges to zero. Can be an int (single axis), or a sequence.
    :param boundaries: numpy array of shape (len(axes), 4). Each row contains the two bounds of the uniform
    distributions from which we draw the width of the zero-bands on each side.
    Those bounds must be expressed in relative side (i.e. between 0 and 1).
    :return: a tensor of the same shape as the input, with bands of zeros along the pecified axes.

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

    def __init__(self, axes, boundaries, **kwargs):
        self.axes = utils.reformat_to_list(axes, dtype='int')
        self.boundaries = utils.reformat_to_n_channels_array(boundaries, n_dims=4, n_channels=len(self.axes))
        self.inputshape = None
        super(MaskEdges, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["axes"] = self.axes
        config["boundaries"] = self.boundaries
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
                                                   maxval=axis_boundaries[3] * self.inputshape[axis]) - idx1)
            idx3 = self.input_shape[axis] - idx1 - idx2
            split_idx = tf.concat([idx1, idx2, idx3], axis=0)

            # update mask
            split_list = tf.split(inputs, split_idx, axis=axis)
            tmp_mask = tf.concat([tf.zeros_like(split_list[0]),
                                  tf.ones_like(split_list[1]),
                                  tf.zeros_like(split_list[2])], axis=axis)
            mask = mask * tmp_mask

        # mask second_channel
        tensor = inputs * mask

        return tensor, mask
