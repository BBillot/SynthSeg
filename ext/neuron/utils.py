"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import itertools
import numpy as np
import tensorflow as tf
import keras.backend as K


def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: an N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc
    """

    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)
    nb_dims = loc.shape[-1]

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')

    if isinstance(vol.shape, tf.TensorShape):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')

        # clip values
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    return interp_vol


def resize(vol, zoom_factor, new_shape, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of length ndims or ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    new_shape should be a list of length ndims

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]
        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)
    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    # get grid for new shape
    grid = volshape_to_ndgrid(new_shape)
    grid = [tf.cast(f, 'float32') for f in grid]
    offset = [grid[f] / zoom_factor[f] - grid[f] for f in range(ndims)]
    offset = tf.stack(offset, ndims)

    # transform
    return transform(vol, offset, interp_method)


zoom = resize


def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)
        indexing

    Returns:
        shift field (Tensor) of size *volshape x N
    """

    if isinstance(volshape, tf.TensorShape):
        volshape = volshape.as_list()

    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) +
                        '%d x %d+1.' % (nb_dims, nb_dims) +
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)


def combine_non_linear_and_aff_to_shift(transform_list, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        transform_list: list of non-linear tensor (size of volshape) and affine ND+1 x ND+1 or ND x ND+1 tensor
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)
        indexing

    Returns:
        shift field (Tensor) of size *volshape x N
    """

    if isinstance(volshape, tf.TensorShape):
        volshape = volshape.as_list()

    # convert transforms to floats
    for i in range(len(transform_list)):
        if transform_list[i].dtype != 'float32':
            transform_list[i] = tf.cast(transform_list[i], 'float32')

    nb_dims = len(volshape)

    # transform affine to matrix if given as vector
    if len(transform_list[1].shape) == 1:
        if len(transform_list[1]) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(transform_list[1]))

        transform_list[1] = tf.reshape(transform_list[1], [nb_dims, nb_dims + 1])

    if not (transform_list[1].shape[0] in [nb_dims, nb_dims + 1] and transform_list[1].shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) +
                        '%d x %d+1.' % (nb_dims, nb_dims) +
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    # non_linear_mesh = tf.unstack(transform_list[0], axis=3)
    non_linear_mesh = tf.unstack(transform_list[0], axis=-1)
    flat_mesh = [flatten(mesh[i]+non_linear_mesh[i]) for i in range(len(mesh))]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # N+1 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(transform_list[1], mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)


def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    transform interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
    
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    """

    # parse shapes
    if isinstance(loc_shift.shape, tf.TensorShape):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be meshed and delta
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return interpn(vol, loc, interp_method=interp_method)


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in tensorflow
    
    Aside from directly using tensorflow's numerical integration odeint(), also implements 
    "scaling and squaring", and quadrature. Note that the diff. equation given to odeint
    is the one used in quadrature.   

    Parameters:
        vec: the Tensor field to integrate. 
            If vol_size is the size of the intrinsic volume, and vol_ndim = len(vol_size),
            then vector shape (vec_shape) should be 
            [vol_size, vol_ndim] (if stationary)
            [vol_size, vol_ndim, nb_time_steps] (if time dependent)
        time_dep: bool whether vector is time dependent
        method: 'scaling_and_squaring' or 'ss' or 'quadrature'
        
        if using 'scaling_and_squaring': currently only supports integrating to time point 1.
            nb_steps int number of steps. Note that this means the vec field gets broken own to 2**nb_steps.
            so nb_steps of 0 means integral = vec.

    Returns:
        int_vec: integral of vector field with same shape as the input
    """

    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2 ** nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2 ** nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])

            disp = svec[0, :]

        else:
            vec = vec / (2 ** nb_steps)
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    else:  # method == 'quadrature':
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    return disp


def volshape_to_ndgrid(volshape, **kwargs):
    """
    compute Tensor ndgrid from a volume size

    Parameters:
        volshape: the volume size

    Returns:
        A list of Tensors

    See Also:
        ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return ndgrid(*linvec, **kwargs)


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors
    
    """
    return meshgrid(*args, indexing='ij', **kwargs)


def meshgrid(*args, **kwargs):
    """
    
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921
    
    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """

    indexing = kwargs.pop("indexing", "xy")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output


def flatten(v):
    """flatten Tensor v"""

    return tf.reshape(v, [-1])


def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs):
    """assumes column-order major"""
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx
