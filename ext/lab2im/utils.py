"""
This file contains all the utilities used in that project. They are classified in 5 categories:
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
    -mkdir
    -mkcmd
4- shape-related functions
    -get_dims
    -get_resample_shape
    -add_axis
    -get_padding_margin
5- build affine matrices/tensors
    -create_affine_transformation_matrix
    -sample_affine_transform
    -create_rotation_transform
    -create_shearing_transform
6- miscellaneous
    -infer
    -LoopInfo
    -get_mapping_lut
    -build_training_generator
    -find_closest_number_divisible_by_m
    -build_binary_structure
    -draw_value_from_distribution
    -build_exp


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


import os
import glob
import math
import time
import pickle
import numpy as np
import nibabel as nib
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from datetime import timedelta
from scipy.ndimage.morphology import distance_transform_edt


# ---------------------------------------------- loading/saving functions ----------------------------------------------


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
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
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        from ext.lab2im import edit_volumes  # the import is done here to avoid import loops
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
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

    mkdir(os.path.dirname(path))
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
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty = nib.Nifti1Image(volume, aff, header)
            nifty.set_data_dtype(dtype)
        else:
            nifty = nib.Nifti1Image(volume, aff, header)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :param max_channels: maximum possible number of channels for the input volume.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        from ext.lab2im import edit_volumes  # the import is done here to avoid import loops
        ras_axes = edit_volumes.get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = edit_volumes.get_ras_axes(aff_ref, n_dims=n_dims)
        im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

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
        # go through all labels files and compute unique list of labels
        labels_paths = list_images_in_folder(labels_dir)
        label_list = np.empty(0)
        loop_info = LoopInfo(len(labels_paths), 10, 'processing', print_time=True)
        for lab_idx, path in enumerate(labels_paths):
            loop_info.update(lab_idx)
            y = load_volume(path, dtype='int32')
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate((label_list, y_unique))).astype('int')

    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                             109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                             251, 252, 253, 254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
                             502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 517, 530,
                             531, 532, 533, 534, 535, 536, 537]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif (0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) | (1000 <= la <= 1035) | \
                    (la == 865) | (20100 < la < 20110):
                if la not in left:
                    left.append(la)
            elif (39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) | (20000 < la < 20010) | (la == 139) | \
                    (la == 866):
                if la not in right:
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
        return np.int32(label_list), None


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
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
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
        return [None] * n_channels
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


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz within a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images


def list_files(path_dir, whole_path=True, expr=None, cond_type='or'):
    """This function returns a list of files contained in a folder, with possible regexp.
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
    """This function returns a list of subfolders contained in a folder, with possible regexp.
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


def get_image_extension(path):
    name = os.path.basename(path)
    if name[-7:] == '.nii.gz':
        return 'nii.gz'
    elif name[-4:] == '.mgz':
        return 'mgz'
    elif name[-4:] == '.nii':
        return 'nii'
    elif name[-4:] == '.npz':
        return 'npz'


def strip_extension(path):
    """Strip classical image extensions (.nii.gz, .nii, .mgz, .npz) from a filename."""
    return path.replace('.nii.gz', '').replace('.nii', '').replace('.mgz', '').replace('.npz', '')


def strip_suffix(path):
    """Strip classical image suffix from a filename."""
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
    path = path.replace('_seg', '')
    path = path.replace('.seg', '')
    path = path.replace('seg', '')
    path = path.replace('_seg_1', '')
    path = path.replace('_seg_2', '')
    path = path.replace('seg_1_', '')
    path = path.replace('seg_2_', '')
    return path


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)


def mkcmd(*args):
    """Creates terminal command with provided inputs.
    Example: mkcmd('mv', 'source', 'dest') will give 'mv source dest'."""
    return ' '.join([str(arg) for arg in args])


# ---------------------------------------------- shape-related functions -----------------------------------------------


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
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
    """Add axis to a numpy array.
    :param x: input array
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


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


# -------------------------------------------- build affine matrices/tensors -------------------------------------------


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """Create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer, can either be 2 or 3.
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


def sample_affine_transform(batchsize,
                            n_dims,
                            rotation_bounds=False,
                            scaling_bounds=False,
                            shearing_bounds=False,
                            translation_bounds=False,
                            enable_90_rotations=False):
    """build batchsize x 4 x 4 tensor representing an affine transformation in homogeneous coordinates.
    If return_inv is True, also returns the inverse of the created affine matrix."""

    if (rotation_bounds is not False) | (enable_90_rotations is not False):
        if n_dims == 2:
            if rotation_bounds is not False:
                rotation = draw_value_from_distribution(rotation_bounds,
                                                        size=1,
                                                        default_range=15.0,
                                                        return_as_tensor=True,
                                                        batchsize=batchsize)
            else:
                rotation = tf.zeros(tf.concat([batchsize, tf.ones(1, dtype='int32')], axis=0))
        else:  # n_dims = 3
            if rotation_bounds is not False:
                rotation = draw_value_from_distribution(rotation_bounds,
                                                        size=n_dims,
                                                        default_range=15.0,
                                                        return_as_tensor=True,
                                                        batchsize=batchsize)
            else:
                rotation = tf.zeros(tf.concat([batchsize, 3 * tf.ones(1, dtype='int32')], axis=0))
        if enable_90_rotations:
            rotation = tf.cast(tf.random.uniform(tf.shape(rotation), maxval=4, dtype='int32') * 90, 'float32') \
                       + rotation
        T_rot = create_rotation_transform(rotation, n_dims)
    else:
        T_rot = tf.tile(tf.expand_dims(tf.eye(n_dims), axis=0),
                        tf.concat([batchsize, tf.ones(2, dtype='int32')], axis=0))

    if shearing_bounds is not False:
        shearing = draw_value_from_distribution(shearing_bounds,
                                                size=n_dims ** 2 - n_dims,
                                                default_range=.01,
                                                return_as_tensor=True,
                                                batchsize=batchsize)
        T_shearing = create_shearing_transform(shearing, n_dims)
    else:
        T_shearing = tf.tile(tf.expand_dims(tf.eye(n_dims), axis=0),
                             tf.concat([batchsize, tf.ones(2, dtype='int32')], axis=0))

    if scaling_bounds is not False:
        scaling = draw_value_from_distribution(scaling_bounds,
                                               size=n_dims,
                                               centre=1,
                                               default_range=.15,
                                               return_as_tensor=True,
                                               batchsize=batchsize)
        T_scaling = tf.linalg.diag(scaling)
    else:
        T_scaling = tf.tile(tf.expand_dims(tf.eye(n_dims), axis=0),
                            tf.concat([batchsize, tf.ones(2, dtype='int32')], axis=0))

    T = tf.matmul(T_scaling, tf.matmul(T_shearing, T_rot))

    if translation_bounds is not False:
        translation = draw_value_from_distribution(translation_bounds,
                                                   size=n_dims,
                                                   default_range=5,
                                                   return_as_tensor=True,
                                                   batchsize=batchsize)
        T = tf.concat([T, tf.expand_dims(translation, axis=-1)], axis=-1)
    else:
        T = tf.concat([T, tf.zeros(tf.concat([tf.shape(T)[:2], tf.ones(1, dtype='int32')], 0))], axis=-1)

    # build rigid transform
    T_last_row = tf.expand_dims(tf.concat([tf.zeros((1, n_dims)), tf.ones((1, 1))], axis=1), 0)
    T_last_row = tf.tile(T_last_row, tf.concat([batchsize, tf.ones(2, dtype='int32')], axis=0))
    T = tf.concat([T, T_last_row], axis=1)

    return T


def create_rotation_transform(rotation, n_dims):
    """build rotation transform from 3d or 2d rotation coefficients. Angles are given in degrees."""
    rotation = rotation * np.pi / 180
    if n_dims == 3:
        shape = tf.shape(tf.expand_dims(rotation[..., 0], -1))

        Rx_row0 = tf.expand_dims(tf.tile(tf.expand_dims(tf.convert_to_tensor([1., 0., 0.]), 0), shape), axis=1)
        Rx_row1 = tf.stack([tf.zeros(shape), tf.expand_dims(tf.cos(rotation[..., 0]), -1),
                            tf.expand_dims(-tf.sin(rotation[..., 0]), -1)], axis=-1)
        Rx_row2 = tf.stack([tf.zeros(shape), tf.expand_dims(tf.sin(rotation[..., 0]), -1),
                            tf.expand_dims(tf.cos(rotation[..., 0]), -1)], axis=-1)
        Rx = tf.concat([Rx_row0, Rx_row1, Rx_row2], axis=1)

        Ry_row0 = tf.stack([tf.expand_dims(tf.cos(rotation[..., 1]), -1), tf.zeros(shape),
                            tf.expand_dims(tf.sin(rotation[..., 1]), -1)], axis=-1)
        Ry_row1 = tf.expand_dims(tf.tile(tf.expand_dims(tf.convert_to_tensor([0., 1., 0.]), 0), shape), axis=1)
        Ry_row2 = tf.stack([tf.expand_dims(-tf.sin(rotation[..., 1]), -1), tf.zeros(shape),
                            tf.expand_dims(tf.cos(rotation[..., 1]), -1)], axis=-1)
        Ry = tf.concat([Ry_row0, Ry_row1, Ry_row2], axis=1)

        Rz_row0 = tf.stack([tf.expand_dims(tf.cos(rotation[..., 2]), -1),
                            tf.expand_dims(-tf.sin(rotation[..., 2]), -1), tf.zeros(shape)], axis=-1)
        Rz_row1 = tf.stack([tf.expand_dims(tf.sin(rotation[..., 2]), -1),
                            tf.expand_dims(tf.cos(rotation[..., 2]), -1), tf.zeros(shape)], axis=-1)
        Rz_row2 = tf.expand_dims(tf.tile(tf.expand_dims(tf.convert_to_tensor([0., 0., 1.]), 0), shape), axis=1)
        Rz = tf.concat([Rz_row0, Rz_row1, Rz_row2], axis=1)

        T_rot = tf.matmul(tf.matmul(Rx, Ry), Rz)

    elif n_dims == 2:
        R_row0 = tf.stack([tf.expand_dims(tf.cos(rotation[..., 0]), -1),
                           tf.expand_dims(tf.sin(rotation[..., 0]), -1)], axis=-1)
        R_row1 = tf.stack([tf.expand_dims(-tf.sin(rotation[..., 0]), -1),
                           tf.expand_dims(tf.cos(rotation[..., 0]), -1)], axis=-1)
        T_rot = tf.concat([R_row0, R_row1], axis=1)

    else:
        raise Exception('only supports 2 or 3D.')

    return T_rot


def create_shearing_transform(shearing, n_dims):
    """build shearing transform from 2d/3d shearing coefficients"""
    shape = tf.shape(tf.expand_dims(shearing[..., 0], -1))
    if n_dims == 3:
        shearing_row0 = tf.stack([tf.ones(shape), tf.expand_dims(shearing[..., 0], -1),
                                  tf.expand_dims(shearing[..., 1], -1)], axis=-1)
        shearing_row1 = tf.stack([tf.expand_dims(shearing[..., 2], -1), tf.ones(shape),
                                  tf.expand_dims(shearing[..., 3], -1)], axis=-1)
        shearing_row2 = tf.stack([tf.expand_dims(shearing[..., 4], -1), tf.expand_dims(shearing[..., 5], -1),
                                  tf.ones(shape)], axis=-1)
        T_shearing = tf.concat([shearing_row0, shearing_row1, shearing_row2], axis=1)

    elif n_dims == 2:
        shearing_row0 = tf.stack([tf.ones(shape), tf.expand_dims(shearing[..., 0], -1)], axis=-1)
        shearing_row1 = tf.stack([tf.expand_dims(shearing[..., 1], -1), tf.ones(shape)], axis=-1)
        T_shearing = tf.concat([shearing_row0, shearing_row1], axis=1)
    else:
        raise Exception('only supports 2 or 3D.')
    return T_shearing


# --------------------------------------------------- miscellaneous ----------------------------------------------------


def infer(x):
    """ Try to parse input to float. If it fails, tries boolean, and otherwise keep it as string """
    try:
        x = float(x)
    except ValueError:
        if x == 'False':
            x = False
        elif x == 'True':
            x = True
        elif not isinstance(x, str):
            raise TypeError('input should be an int/float/boolean/str, had {}'.format(type(x)))
    return x


class LoopInfo:
    """
    Class to print the current iteration in a for loop, and optionally the estimated remaining time.
    Instantiate just before the loop, and call the update method at the start of the loop.
    The printed text has the following format:
    processing i/total    remaining time: hh:mm:ss
    """

    def __init__(self, n_iterations, spacing=10, text='processing', print_time=False):
        """
        :param n_iterations: total number of iterations of the for loop.
        :param spacing: frequency at which the update info will be printed on screen.
        :param text: text to print. Default is processing.
        :param print_time: whether to print the estimated remaining time. Default is False.
        """

        # loop parameters
        self.n_iterations = n_iterations
        self.spacing = spacing

        # text parameters
        self.text = text
        self.print_time = print_time
        self.print_previous_time = False
        self.align = len(str(self.n_iterations)) * 2 + 1 + 3

        # timing parameters
        self.iteration_durations = np.zeros((n_iterations,))
        self.start = time.time()
        self.previous = time.time()

    def update(self, idx):

        # time iteration
        now = time.time()
        self.iteration_durations[idx] = now - self.previous
        self.previous = now

        # print text
        if idx == 0:
            print(self.text + ' 1/{}'.format(self.n_iterations))
        elif idx % self.spacing == self.spacing - 1:
            iteration = str(idx + 1) + '/' + str(self.n_iterations)
            if self.print_time:
                # estimate remaining time
                max_duration = np.max(self.iteration_durations)
                average_duration = np.mean(self.iteration_durations[self.iteration_durations > .01 * max_duration])
                remaining_time = int(average_duration * (self.n_iterations - idx))
                # print total remaining time only if it is greater than 1s or if it was previously printed
                if (remaining_time > 1) | self.print_previous_time:
                    eta = str(timedelta(seconds=remaining_time))
                    print(self.text + ' {:<{x}} remaining time: {}'.format(iteration, eta, x=self.align))
                    self.print_previous_time = True
                else:
                    print(self.text + ' {}'.format(iteration))
            else:
                print(self.text + ' {}'.format(iteration))


def get_mapping_lut(source, dest=None):
    """This functions returns the look-up table to map a list of N values (source) to another list (dest).
    If the second list is not given, we assume it is equal to [0, ..., N-1]."""

    # initialise
    source = np.array(reformat_to_list(source), dtype='int32')
    n_labels = source.shape[0]

    # build new label list if necessary
    if dest is None:
        dest = np.arange(n_labels, dtype='int32')
    else:
        assert len(source) == len(dest), 'label_list and new_label_list should have the same length'
        dest = np.array(reformat_to_list(dest, dtype='int'))

    # build look-up table
    lut = np.zeros(np.max(source) + 1, dtype='int32')
    for source, dest in zip(source, dest):
        lut[source] = dest

    return lut


def build_training_generator(gen, batchsize):
    """Build generator for training a network."""
    while True:
        inputs = next(gen)
        if batchsize > 1:
            target = np.concatenate([np.zeros((1, 1))] * batchsize, 0)
        else:
            target = np.zeros((1, 1))
        yield inputs, target


def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)


def build_binary_structure(connectivity, n_dims, shape=None):
    """Return a dilation/erosion element with provided connectivity"""
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


def draw_value_from_distribution(hyperparameter,
                                 size=1,
                                 distribution='uniform',
                                 centre=0.,
                                 default_range=10.0,
                                 positive_only=False,
                                 return_as_tensor=False,
                                 batchsize=None):
    """Sample values from a uniform, or normal distribution of given hyperparameters.
    These hyperparameters are to the number of 2 in both uniform and normal cases.
    :param hyperparameter: values of the hyperparameters. Can either be:
    1) None, in each case the two hyperparameters are given by [center-default_range, center+default_range],
    2) a number, where the two hyperparameters are given by [centre-hyperparameter, centre+hyperparameter],
    3) a sequence of length 2, directly defining the two hyperparameters: [min, max] if the distribution is uniform,
    [mean, std] if the distribution is normal.
    4) a numpy array, with size (2, m). In this case, the function returns a 1d array of size m, where each value has
    been sampled independently with the specified hyperparameters. If the distribution is uniform, rows correspond to
    its lower and upper bounds, and if the distribution is normal, rows correspond to its mean and std deviation.
    5) a numpy array of size (2*n, m). Same as 4) but we first randomly select a block of two rows among the
    n possibilities.
    6) the path to a numpy array corresponding to case 4 or 5.
    7) False, in which case this function returns None.
    :param size: (optional) number of values to sample. All values are sampled independently.
    Used only if hyperparameter is not a numpy array.
    :param distribution: (optional) the distribution type. Can be 'uniform' or 'normal'. Default is 'uniform'.
    :param centre: (optional) default centre to use if hyperparameter is None or a number.
    :param default_range: (optional) default range to use if hyperparameter is None.
    :param positive_only: (optional) whether to reset all negative values to zero.
    :param return_as_tensor: (optional) whether to return the result as a tensorflow tensor
    :param batchsize: (optional) if return_as_tensor is true, then you can sample a tensor of a given batchsize. Give
    this batchsize as a tensorflow tensor here.
    :return: a float, or a numpy 1d array if size > 1, or hyperparameter is itself a numpy array.
    Returns None if hyperparameter is False.
    """

    # return False is hyperparameter is False
    if hyperparameter is False:
        return None

    # reformat parameter_range
    hyperparameter = load_array_if_path(hyperparameter, load_as_numpy=True)
    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array([[centre - default_range] * size, [centre + default_range] * size])
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array([[centre - hyperparameter] * size, [centre + hyperparameter] * size])
        elif isinstance(hyperparameter, (list, tuple)):
            assert len(hyperparameter) == 2, 'if list, parameter_range should be of length 2.'
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError('parameter_range should either be None, a number, a sequence, or a numpy array.')
    elif isinstance(hyperparameter, np.ndarray):
        assert hyperparameter.shape[0] % 2 == 0, 'number of rows of parameter_range should be divisible by 2'
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx: modality_idx + 2, :]

    # draw values as tensor
    if return_as_tensor:
        shape = KL.Lambda(lambda x: tf.convert_to_tensor(hyperparameter.shape[1], 'int32'))([])
        if batchsize is not None:
            shape = KL.Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], axis=0)], axis=0))([batchsize, shape])
        if distribution == 'uniform':
            parameter_value = KL.Lambda(lambda x: tf.random.uniform(shape=x,
                                                                    minval=hyperparameter[0, :],
                                                                    maxval=hyperparameter[1, :]))(shape)
        elif distribution == 'normal':
            parameter_value = KL.Lambda(lambda x: tf.random.normal(shape=x,
                                                                   mean=hyperparameter[0, :],
                                                                   stddev=hyperparameter[1, :]))(shape)
        else:
            raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

        if positive_only:
            parameter_value = KL.Lambda(lambda x: K.clip(x, 0, None))(parameter_value)

    # draw values as numpy array
    else:
        if distribution == 'uniform':
            parameter_value = np.random.uniform(low=hyperparameter[0, :], high=hyperparameter[1, :])
        elif distribution == 'normal':
            parameter_value = np.random.normal(loc=hyperparameter[0, :], scale=hyperparameter[1, :])
        else:
            raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

        if positive_only:
            parameter_value[parameter_value < 0] = 0

    return parameter_value


def build_exp(x, first, last, fix_point):
    # first = f(0), last = f(+inf), fix_point = [x0, f(x0))]
    a = last
    b = first - last
    c = - (1 / fix_point[0]) * np.log((fix_point[1] - last) / (first - last))
    return a + b * np.exp(-c * x)
