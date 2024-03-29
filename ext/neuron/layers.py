"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/integration functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from copy import deepcopy

# local
from ext.neuron.utils import transform, resize, integrate_vec, affine_to_shift, combine_non_linear_and_aff_to_shift


class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms. 
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from 
    the identity matrix.

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform
        self.is_affine = list()

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["interp_method"] = self.interp_method
        config["indexing"] = self.indexing
        config["single_transform"] = self.single_transform
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: list of transform Tensors
            if affine:
                should be an N+1 x N+1 matrix
                *or* a N+1*N+1 tensor (which will be reshaped to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 3:
            raise Exception('Spatial Transformer must be called on a list of min length 2 and max length 3.'
                            'First argument is the image followed by the affine and non linear transforms.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        trf_shape = [trans_shape[1:] for trans_shape in input_shape[1:]]

        for (i, shape) in enumerate(trf_shape):

            # the transform is an affine iff:
            # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
            # it's a 2D Tensor and shape == [N+1, N+1].
            self.is_affine.append(len(shape) == 1 or
                                  (len(shape) == 2 and all([f == (self.ndims + 1) for f in shape])))

            # check sizes
            if self.is_affine[i] and len(shape) == 1:
                ex = self.ndims * (self.ndims + 1)
                if shape[0] != ex:
                    raise Exception('Expected flattened affine of len %d but got %d' % (ex, shape[0]))

            if not self.is_affine[i]:
                if shape[-1] != self.ndims:
                    raise Exception('Offset flow field size expected: %d, found: %d' % (self.ndims, shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Parameters
            inputs: list with several entries: the volume followed by the transforms
        """

        # check shapes
        assert 1 < len(inputs) < 4, "inputs has to be len 2 or 3, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1:]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        for i in range(len(trf)):
            trf[i] = K.reshape(trf[i], [-1, *self.inshape[i+1][1:]])

        # reorder transforms, non-linear first and affine second
        ind_nonlinear_linear = [i[0] for i in sorted(enumerate(self.is_affine), key=lambda x:x[1])]
        self.is_affine = [self.is_affine[i] for i in ind_nonlinear_linear]
        self.inshape = [self.inshape[i] for i in ind_nonlinear_linear]
        trf = [trf[i] for i in ind_nonlinear_linear]

        # go from affine to deformation field
        if len(trf) == 1:
            trf = trf[0]
            if self.is_affine[0]:
                trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)
        # combine non-linear and affine to obtain a single deformation field
        elif len(trf) == 2:
            trf = tf.map_fn(lambda x: self._non_linear_and_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(self._single_transform, [vol, trf[0, :]], dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])
        return affine_to_shift(trf, volshape, shift_center=True)

    def _non_linear_and_aff_to_shift(self, trf, volshape):
        if len(trf[1].shape) == 1:  # go from vector to matrix
            trf[1] = tf.reshape(trf[1], [self.ndims, self.ndims + 1])
        return combine_non_linear_and_aff_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method)


class VecInt(Layer):
    """
    Vector Integration Layer

    Enables vector integration via several methods 
    (ode or quadrature for time-dependent vector fields, 
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, indexing='ij', method='ss', int_steps=7, out_time_pt=1,
                 ode_args=None,
                 odeint_fn=None, **kwargs):
        """        
        Parameters:
            method can be any of the methods in neuron.utils.integrate_vec
            indexing can be 'xy' (switches first two dimensions) or 'ij'
            int_steps is the number of integration steps
            out_time_pt is time point at which to output if using odeint integration
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["indexing"] = self.indexing
        config["method"] = self.method
        config["int_steps"] = self.int_steps
        config["out_time_pt"] = self.out_time_pt
        config["ode_args"] = self.ode_args
        config["odeint_fn"] = self.odeint_fn
        return config

    def build(self, input_shape):
        # confirm built
        self.built = True

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d' % (trf_shape[-1], len(trf_shape) - 2))

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi_gpu models...
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        if len(inputs) > 1:
            assert self.out_time_pt is None, 'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out = tf.map_fn(self._single_int, [loc_shift] + inputs[1:], dtype=tf.float32)
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                             nb_steps=self.int_steps,
                             ode_args=self.ode_args,
                             out_time_pt=out_time_pt,
                             odeint_fn=self.odeint_fn)


class Resize(Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this function useful, please cite:
    Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,Dalca AV, Guttag J, Sabuncu MR
    CVPR 2018

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 zoom_factor=None,
                 size=None,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.size = list(size)
        self.zoom_factor0 = None
        self.size0 = None
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["zoom_factor"] = self.zoom_factor
        config["size"] = self.size
        config["interp_method"] = self.interp_method
        return config

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape

        # check zoom_factor
        if isinstance(self.zoom_factor, float):
            self.zoom_factor0 = [self.zoom_factor] * self.ndims
        elif self.zoom_factor is None:
            self.zoom_factor0 = [0] * self.ndims
        elif isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor0 = deepcopy(self.zoom_factor)
            assert len(self.zoom_factor0) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'.format(len(self.zoom_factor), self.ndims)
        else:
            raise Exception('zoom_factor should be an int or a list/tuple of int (or None if size is not set to None)')

        # check size
        if isinstance(self.size, int):
            self.size0 = [self.size] * self.ndims
        elif self.size is None:
            self.size0 = [0] * self.ndims
        elif isinstance(self.size, (list, tuple)):
            self.size0 = deepcopy(self.size)
            assert len(self.size0) == self.ndims, \
                'size length {} does not match number of dimensions {}'.format(len(self.size0), self.ndims)
        else:
            raise Exception('size should be an int or a list/tuple of int (or None if zoom_factor is not set to None)')

        # confirm built
        self.built = True

        super(Resize, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        """
        Parameters
            inputs: volume or list of one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[1:]])

        # set value of missing size or zoom_factor
        if not any(self.zoom_factor0):
            self.zoom_factor0 = [self.size0[i] / self.inshape[i+1] for i in range(self.ndims)]
        else:
            self.size0 = [int(self.inshape[f+1] * self.zoom_factor0[f]) for f in range(self.ndims)]

        # map transform across batch
        return tf.map_fn(self._single_resize, vol, dtype=vol.dtype)

    def compute_output_shape(self, input_shape):

        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] * self.zoom_factor0[f]) for f in range(self.ndims)]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return resize(inputs, self.zoom_factor0, self.size0, interp_method=self.interp_method)


# Zoom naming of resize, to match scipy's naming
Zoom = Resize


#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################

class LocalBias(Layer):
    """ 
    Local bias layer: each pixel/voxel has its own bias operation (one parameter)
    out[v] = in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        self.kernel = None
        super(LocalBias, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["my_initializer"] = self.initializer
        config["biasmult"] = self.biasmult
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return x + self.kernel * self.biasmult  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape
