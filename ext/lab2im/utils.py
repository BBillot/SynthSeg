"""This file contains all the utilities used in that project. They are classified in 5 categories:
1- loading/saving functions:
    -load_volume
    -save_volume
    -get_volume_info
    -get_list_labels
    -load_array_if_path
    -write_pickle
    -read_pickle
    -write_model_summary
2- reformatting functions
    -reformat_to_list
    -reformat_to_n_channels_array
3- path related functions
    -list_images_in_folder
    -list_files
    -list_subfolders
    -strip_extension
    -strip_suffix
4- shape-related functions
    -get_dims
    -get_resample_shape
    -add_axis
    -get_padding_margin
5- miscellaneous
    -print_loop_info
    -rearrange_label_list
    -build_training_generator
    -find_closest_number_divisible_by_m
    -build_binary_structure
    -build_gaussian_kernel
    -get_std_blurring_mask_for_downsampling
    -draw_value_from_distribution
    -create_affine_transformation_matrix
"""

import os
import glob
import math
import pickle
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import distance_transform_edt


# ---------------------------------------------- loading/saving functions ----------------------------------------------


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_data())
        else:
            volume = x.get_data()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        from . import edit_volumes  # the import is done here to avoid import loops
        n_dims, _ = get_dims(list(volume.shape), max_channels=3)
        volume, aff = edit_volumes.align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """
    if dtype is not None:
        volume = volume.astype(dtype=dtype)
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :return: volume (if return_volume is true), and corresponding info.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=3)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii.gz' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1]).tolist()
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta']).tolist()  # mgz image
    else:
        data_res = [1.0] * n_dims

    # align to given affine matrix
    if aff_ref is not None:
        from . import edit_volumes  # the import is done here to avoid import loops
        ras_axes = edit_volumes.get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = edit_volumes.get_ras_axes(aff_ref, n_dims=n_dims)
        im, aff = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()
        data_res = data_res.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def get_list_labels(label_list=None, labels_dir=None, save_label_list=None, FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list. Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: (optional) if path_label_list is None, the label list is computed by reading all the label maps
    in the given folder. Can also be the path to a single label map.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    if label_list is not None:
        label_list = np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int'))

    # compute label list from all label files
    elif labels_dir is not None:
        print('Compiling list of unique labels')
        # prepare data files
        if ('.nii.gz' in labels_dir) | ('.nii' in labels_dir) | ('.mgz' in labels_dir) | ('.npz' in labels_dir):
            labels_paths = [labels_dir]
        else:
            labels_paths = list_images_in_folder(labels_dir)
        assert len(labels_paths) > 0, "Could not find any training data"
        # go through all labels files and compute unique list of labels
        label_list = np.empty(0)
        for lab_idx, path in enumerate(labels_paths):
            print_loop_info(lab_idx, len(labels_paths), 10)
            y = load_volume(path)
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate((label_list, y_unique))).astype('int')

    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 165, 251, 252, 253, 254, 255, 258, 259,
                             331, 332, 333, 334, 335, 336, 337, 338, 339, 340]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (20100 < la < 20110):
                left.append(la)
            elif (39 < la < 72) | (20000 < la < 20010):
                right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
        return np.int32(label_list)


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


def write_pickle(filepath, obj):
    """ write a python object with a pickle at a given path"""
    with open(filepath, 'wb') as file:
        pickler = pickle.Pickler(file)
        pickler.dump(obj)


def read_pickle(filepath):
    """ read a python object with a pickle"""
    with open(filepath, 'rb') as file:
        unpickler = pickle.Unpickler(file)
        return unpickler.load()


def write_model_summary(model, filepath='./model_summary.txt', line_length=150):
    """Write the summary of a keras model at a given path, with a given length for each line"""
    with open(filepath, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=line_length)


# ----------------------------------------------- reformatting functions -----------------------------------------------


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var


def reformat_to_n_channels_array(var, n_dims=3, n_channels=1):
    """This function takes an int, float, list or tuple and reformat it to an array of shape (n_channels, n_dims).
    If resolution is a str, it will be assumed to be the path of a numpy array.
    If resolution is a numpy array, it will be checked to have shape (n_channels, n_dims).
    Finally if resolution is None, this function returns None as well."""
    if var is None:
        return None
    if isinstance(var, str):
        var = np.load(var)
    # convert to numpy array
    if isinstance(var, (int, float, list, tuple)):
        var = reformat_to_list(var, n_dims)
        var = np.tile(np.array(var), (n_channels, 1))
    # check shape if numpy array
    elif isinstance(var, np.ndarray):
        if n_channels == 1:
            var = var.reshape((1, n_dims))
        else:
            if np.squeeze(var).shape == (n_dims,):
                var = np.tile(var.reshape((1, n_dims)), (n_channels, 1))
            elif var.shape != (n_channels, n_dims):
                raise ValueError('if array, var should be {0} or {1}'.format((1, n_dims), (n_channels, n_dims)))
    else:
        raise TypeError('var should be int, float, list, tuple or ndarray')
    return np.round(var, 3)


# ----------------------------------------------- path-related functions -----------------------------------------------


def list_images_in_folder(path_dir):
    """List all files with extension nii, nii.gz, mgz, or npz whithin a folder."""
    list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                         glob.glob(os.path.join(path_dir, '*nii')) +
                         glob.glob(os.path.join(path_dir, '*.mgz')) +
                         glob.glob(os.path.join(path_dir, '*.npz')))
    assert len(list_images) > 0, 'no nii, nii.gz, mgz or npz could be found in %s' % path_dir
    return list_images


def list_files(path_dir, whole_path=True, expr=None, cond_type='or'):
    """This function returns a list of files contained in a folder, whith possible regexp.
    :param path_dir: path of a folder
    :param whole_path: (optional) whether to return whole path or just the filenames.
    :param expr: (optional) regexp for files to list. Can be a str or a list of str.
    :param cond_type: (optional) if exp is a list, specify the logical link between expressions in exp.
    Can be 'or', or 'and'.
    :return: a list of files
    """
    assert isinstance(whole_path, bool), "whole_path should be bool"
    assert cond_type in ['or', 'and'], "cond_type should be either 'or', or 'and'"
    if whole_path:
        files_list = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir)
                             if os.path.isfile(os.path.join(path_dir, f))])
    else:
        files_list = sorted([f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))])
    if expr is not None:  # assumed to be either str or list of str
        if isinstance(expr, str):
            expr = [expr]
        elif not isinstance(expr, (list, tuple)):
            raise Exception("if specified, 'expr' should be a string or list of strings.")
        matched_list_files = list()
        for match in expr:
            tmp_matched_files_list = sorted([f for f in files_list if match in os.path.basename(f)])
            if cond_type == 'or':
                files_list = [f for f in files_list if f not in tmp_matched_files_list]
                matched_list_files += tmp_matched_files_list
            elif cond_type == 'and':
                files_list = tmp_matched_files_list
                matched_list_files = tmp_matched_files_list
        files_list = sorted(matched_list_files)
    return files_list


def list_subfolders(path_dir, whole_path=True, expr=None, cond_type='or'):
    """This function returns a list of subfolders contained in a folder, whith possible regexp.
    :param path_dir: path of a folder
    :param whole_path: (optional) whether to return whole path or just the subfolder names.
    :param expr: (optional) regexp for files to list. Can be a str or a list of str.
    :param cond_type: (optional) if exp is a list, specify the logical link between expressions in exp.
    Can be 'or', or 'and'.
    :return: a list of subfolders
    """
    assert isinstance(whole_path, bool), "whole_path should be bool"
    assert cond_type in ['or', 'and'], "cond_type should be either 'or', or 'and'"
    if whole_path:
        subdirs_list = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir)
                               if os.path.isdir(os.path.join(path_dir, f))])
    else:
        subdirs_list = sorted([f for f in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, f))])
    if expr is not None:  # assumed to be either str or list of str
        if isinstance(expr, str):
            expr = [expr]
        elif not isinstance(expr, (list, tuple)):
            raise Exception("if specified, 'expr' should be a string or list of strings.")
        matched_list_subdirs = list()
        for match in expr:
            tmp_matched_list_subdirs = sorted([f for f in subdirs_list if match in os.path.basename(f)])
            if cond_type == 'or':
                subdirs_list = [f for f in subdirs_list if f not in tmp_matched_list_subdirs]
                matched_list_subdirs += tmp_matched_list_subdirs
            elif cond_type == 'and':
                subdirs_list = tmp_matched_list_subdirs
                matched_list_subdirs = tmp_matched_list_subdirs
        subdirs_list = sorted(matched_list_subdirs)
    return subdirs_list


def strip_extension(path):
    """Strip classical image extensions (.nii.gz, .nii, .mgz, .npz) from a filename."""
    path = path.replace('.nii.gz', '')
    path = path.replace('.nii', '')
    path = path.replace('.mgz', '')
    path = path.replace('.npz', '')
    return path


def strip_suffix(path):
    """Strip classical image suffix from a filename."""
    path = path.replace('_seg', '')
    path = path.replace('.seg', '')
    path = path.replace('seg', '')
    path = path.replace('_seg_1', '')
    path = path.replace('_seg_2', '')
    path = path.replace('seg_1_', '')
    path = path.replace('seg_2_', '')
    path = path.replace('_aseg', '')
    path = path.replace('aseg', '')
    path = path.replace('.aseg', '')
    path = path.replace('_aseg_1', '')
    path = path.replace('_aseg_2', '')
    path = path.replace('aseg_1_', '')
    path = path.replace('aseg_2_', '')
    path = path.replace('_orig', '')
    path = path.replace('orig', '')
    path = path.replace('.orig', '')
    path = path.replace('_norm', '')
    path = path.replace('norm', '')
    path = path.replace('.norm', '')
    path = path.replace('_talairach', '')
    path = path.replace('GSP_FS_4p5', 'GSP')
    path = path.replace('.nii_crispSegmentation', '')
    path = path.replace('_crispSegmentation', '')
    return path


# ---------------------------------------------- shape-related functions -----------------------------------------------


def get_dims(shape, max_channels=3):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=3) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=3) = (3, 3)
    example 3: get_dims([150, 150, 150, 5], max_channels=10) = (3, 5)"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_resample_shape(patch_shape, factor, n_channels=None):
    """Compute the shape of a resampled array given a shape factor.
    :param patch_shape: size of the initial array (without number of channels).
    :param factor: resampling factor. Can be a number, sequence, or 1d numpy array.
    :param n_channels: (optional) if not None, add a number of channel at the end of the computed shape.
    :return: list containing the shape of the input array after being resampled by the given factor.
    """
    factor = reformat_to_list(factor, length=len(patch_shape))
    shape = [math.ceil(patch_shape[i] * factor[i]) for i in range(len(patch_shape))]
    if n_channels is not None:
        shape += [n_channels]
    return shape


def add_axis(x, axis=0):
    """Add axis to a numpy array. The new axis can be added to the first dimension (axis=0), to the last dimension
    (axis=-1), or to both (axis=-2)."""
    if axis == 0:
        return x[np.newaxis, ...]
    elif axis == -1:
        return x[..., np.newaxis]
    elif axis == -2:
        return x[np.newaxis, ..., np.newaxis]
    else:
        raise Exception('axis should be 0 (first), -1 (last), or -2 (first and last)')


def get_padding_margin(cropping, loss_cropping):
    """Compute padding margin"""
    if (cropping is not None) & (loss_cropping is not None):
        cropping = reformat_to_list(cropping)
        loss_cropping = reformat_to_list(loss_cropping)
        n_dims = max(len(cropping), len(loss_cropping))
        cropping = reformat_to_list(cropping, length=n_dims)
        loss_cropping = reformat_to_list(loss_cropping, length=n_dims)
        padding_margin = [int((cropping[i] - loss_cropping[i]) / 2) for i in range(n_dims)]
        if len(padding_margin) == 1:
            padding_margin = padding_margin[0]
    else:
        padding_margin = None
    return padding_margin


# --------------------------------------------------- miscellaneous ----------------------------------------------------


def print_loop_info(idx, n_iterations, spacing=None):
    """Print loop iteration number.
    :param idx: iteration number
    :param n_iterations: total number iterations.
    :param spacing: frequency at which to print loop advancement.
    """
    if idx == 0:
        print('processing {}/{}'.format(1, n_iterations))
    elif idx % spacing == spacing - 1:
        print('processing {}/{}'.format(idx + 1, n_iterations))


def rearrange_label_list(label_list):
    """This functions maps a list of N values between 0 and N-1, and gives the corresponding look-up table."""
    label_list = np.array(reformat_to_list(label_list))
    n_labels = label_list.shape[0]
    new_label_list = np.arange(n_labels)
    lut = np.zeros(np.max(label_list).astype('int') + 1)
    for n in range(n_labels):
        lut[label_list[n].astype('int')] = n
    return new_label_list, lut


def build_training_generator(gen, batchsize):
    """Build generator for training a network."""
    while True:
        inputs = next(gen)
        if batchsize > 1:
            target = np.concatenate([add_axis(np.zeros(1))] * batchsize, 0)
        else:
            target = add_axis(np.zeros(1))
        yield inputs, target


def find_closest_number_divisible_by_m(n, m, smaller_ans=True):
    """Return the closest integer to n that is divisible by m.
    If smaller_ans is True, only values lower than n are considered."""
    # quotient
    q = int(n / m)
    # 1st possible closest number
    n1 = m * q
    # 2nd possible closest number
    if (n * m) > 0:
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))
    # find closest solution
    if (abs(n - n1) < abs(n - n2)) | smaller_ans:
        return n1
    else:
        return n2


def build_binary_structure(connectivity, n_dims):
    """Return a dilation/erosion element with provided connectivity"""
    shape = [connectivity * 2 + 1] * n_dims
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


def build_gaussian_kernel(sigma, n_dims):
    """This function builds a gaussian kernel of specified std deviation for a given number of dimensions.
    :param sigma: standard deviation. Can be a number, a sequence or a 1d numpy array.
    :param n_dims: number of dimension for the returned Gaussian kernel.
    :return: a gaussian kernel of dimension n_dims and of specified std deviation in each direction.
    """
    sigma = reformat_to_list(sigma, length=n_dims, dtype='float')
    shape = [math.ceil(2.5*s) for s in sigma]
    shape = [s + 1 if s % 2 == 0 else s for s in shape]
    if n_dims == 2:
        m, n = [(ss-1.)/2. for ss in shape]
        x, y = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x/(sigma[0]**2) + y*y/(sigma[1]**2)) / 2)
    elif n_dims == 3:
        m, n, p = [(ss-1.)/2. for ss in shape]
        x, y, z = np.ogrid[-m:m+1, -n:n+1, -p:p+1]
        h = np.exp(-(x*x/(sigma[0]**2) + y*y/(sigma[1])**2 + z*z/(sigma[2]**2)) / 2)
    else:
        raise Exception('dimension > 3 not supported')
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_std_blurring_mask_for_downsampling(dowsample_res, current_res, thickness=None):
    """Compute standard deviations of 1d gaussian masks for image blurring before downsampling.
    :param dowsample_res: resolution to dowsample to. Can be a 1d numpy array or list.
    :param current_res: resolution of the volume before downsampling.
    Can be a 1d numpy array or list of the same length as downsample res.
    :param thickness: slices thickness in each dimension.
    Can be a 1d numpy array or list of the same length as downsample res.
    :return: standard deviation of the blurring masks
    """
    # reformat data resolution at which we blur
    n_dims = len(dowsample_res)
    if thickness is not None:
        dowsample_res = [min(dowsample_res[i], thickness[i]) for i in range(n_dims)]

    # build 1d blurring kernels for each direction
    sigma = [0] * n_dims
    for i in range(n_dims):
        # define sigma
        if dowsample_res[i] == 0:
            sigma[i] = 0
        elif current_res[i] == dowsample_res[i]:
            sigma[i] = np.float32(0.5)
        else:
            sigma[i] = np.float32(0.75 * np.around(dowsample_res[i] / current_res[i], 3))

    return sigma


def draw_value_from_distribution(hyperparameter,
                                 size=1,
                                 distribution='uniform',
                                 centre=0.,
                                 default_range=10.0,
                                 positive_only=False):
    """Sample values from a uniform, or normal distribution of given hyper-parameters.
    These hyper-parameters are to the number of 2 in both uniform and normal cases.
    :param hyperparameter: values of the hyper-parameters. Can either be:
    1) None, in each case the two hyper-parameters are given by [center-default_range, center+default_range],
    2) a number, where the two hyper-parameters are given by [centre-hyperparameter, centre+hyperparameter],
    3) a sequence of length 2, directly defining the two hyper-parameters: [min, max] if the distribution is uniform,
    [mean, std] if the distribution is normal.
    4) a numpy array, with size (2, m). In this case, the function returns a 1d array of size m, where each value has
    been sampled independently with the specified hyper-parameters. If the distribution is uniform, rows correspond to
    its lower and upper bounds, and if the distribution is normal, rows correspond to its mean and std deviation.
    5) a numpy array of size (2*n, m). Same as 4) but we first randomly select a block of two rows among the
    n possibilities.
    :param size: (optional) number of values to sample. All values are sampled independently.
    Used only if hyperparameter is not a numpy array.
    :param distribution: (optional) the distribution type. Can be 'uniform' or 'normal'. Default is 'uniform'.
    :param centre: (optional) default centre to use if hyperparameter is None or a number.
    :param default_range: (optional) default range to use if hyperparameter is None.
    :param positive_only: (optional) wheter to reset all negative values to zero.
    :return: a float, or a numpy 1d array if size > 1, or hyperparameter is itself a numpy array.
    """

    # reformat parameter_range
    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array([[centre - default_range] * size, [centre + default_range] * size])
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array([[centre - hyperparameter] * size, [centre + hyperparameter] * size])
        elif isinstance(hyperparameter, (list, tuple)):
            assert len(hyperparameter) == 2, 'if list, parameter_range should be of length 2.'
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError('parameter_range should either be None, a nummber, a sequence, or a numpy array.')
    elif isinstance(hyperparameter, np.ndarray):
        assert hyperparameter.shape[0] % 2 == 0, 'number of rows of parameter_range should be divisible by 2'
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx: modality_idx + 2, :]

    # draw values
    if distribution == 'uniform':
        parameter_value = np.random.uniform(low=hyperparameter[0, :], high=hyperparameter[1, :])
    elif distribution == 'normal':
        parameter_value = np.random.normal(loc=hyperparameter[0, :], scale=hyperparameter[1, :])
    else:
        raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

    if positive_only:
        parameter_value[parameter_value < 0] = 0

    return parameter_value


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """Create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    T_scaling = np.eye(n_dims + 1)
    T_shearing = np.eye(n_dims + 1)
    T_translation = np.eye(n_dims + 1)

    if scaling is not None:
        T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        T_shearing[shearing_index] = shearing

    if translation is not None:
        T_translation[np.arange(n_dims), n_dims * np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot = np.eye(n_dims + 1)
        T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[0]), np.sin(rotation[0]),
                                                                 np.sin(rotation[0]) * -1, np.cos(rotation[0])]
        return T_translation @ T_rot @ T_shearing @ T_scaling

    else:

        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot1 = np.eye(n_dims + 1)
        T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]), np.sin(rotation[0]),
                                                                  np.sin(rotation[0]) * -1, np.cos(rotation[0])]
        T_rot2 = np.eye(n_dims + 1)
        T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]), np.sin(rotation[1]) * -1,
                                                                  np.sin(rotation[1]), np.cos(rotation[1])]
        T_rot3 = np.eye(n_dims + 1)
        T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]), np.sin(rotation[2]),
                                                                  np.sin(rotation[2]) * -1, np.cos(rotation[2])]
        return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling
