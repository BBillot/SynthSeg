"""This file contains functions to edit/preprocess volumes (i.e. not tensors!).
These functions are sorted in five categories:
1- volume editting: this can be applied to any volume (i.e. images or label maps). It contains:
        -mask_volume
        -rescale_volume
        -crop_volume
        -crop_volume_around_region
        -crop_volume_with_idx
        -pad_volume
        -flip_volume
        -get_ras_axes_and_signs
        -align_volume_to_ref
        -blur_volume
2- label map editting: can be applied to label maps only. It contains:
        -correct_label_map
        -mask_label_map
        -smooth_label_map
        -erode_label_map
        -compute_hard_volumes
        -compute_distance_map
3- editting all volumes in a folder: functions are more or less the same as 1, but they now apply to all the volumes
in a given folder. Thus we provide folder paths rather than numpy arrays as inputs. It contains:
        -mask_images_in_dir
        -rescale_images_in_dir
        -crop_images_in_dir
        -crop_images_around_region_in_dir
        -pad_images_in_dir
        -flip_images_in_dir
        -align_images_in_dir
        -correct_nans_images_in_dir
        -blur_images_in_dir
        -create_mutlimodal_images
        -convert_images_in_dir_to_nifty
        -mri_convert_images_in_dir
        -samseg_images_in_dir
        -simulate_upsampled_anisotropic_images
        -check_images_in_dir
4- label maps in dir: same as 3 but for label map-specific functions. It contains:
        -correct_labels_in_dir
        -mask_labels_in_dir
        -smooth_labels_in_dir
        -erode_labels_in_dir
        -upsample_labels_in_dir
        -compute_hard_volumes_in_dir
        -build_atlas
5- dataset editting: functions for editting datasets (i.e. images with corresponding label maps). It contains:
        -check_images_and_labels
        -crop_dataset_to_minimum_size
        -subdivide_dataset_to_patches"""


# python imports
import os
import csv
import shutil
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

# project imports
from . import utils
from .edit_tensors import get_gaussian_1d_kernels, blur_tensor, convert_labels


# ---------------------------------------------------- edit volume -----------------------------------------------------

def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False, masking_value=0,
                return_mask=False):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be uni- or multi-channel.
     In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    vol_shape = list(volume.shape)
    n_dims, n_channels = utils.get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], 'mask should have shape {0}, or {1}, had {2}'.format(
            vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape))
        mask = mask > 0
    if dilate > 0:
        dilate_struct = utils.build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = utils.build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    mask_to_apply = mask | mask_to_apply
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == volume.shape:
        volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return volume, mask_to_apply
    else:
        return volume


def rescale_volume(volume, new_min=0, new_max=255, min_percentile=0.02, max_percentile=0.98, use_positive_only=True):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume
    :param max_percentile: (optional) percentile for estimating robust maximum of volume
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # sort intensities
    if use_positive_only:
        intensities = np.sort(volume[volume > 0])
    else:
        intensities = np.sort(volume.flatten())

    # define robust max and min
    robust_min = np.maximum(0, intensities[int(intensities.shape[0] * min_percentile)])
    robust_max = intensities[int(intensities.shape[0] * max_percentile)]

    # trim values outside range
    volume = np.clip(volume, robust_min, robust_max)

    # rescale image
    volume = new_min + (volume-robust_min) / (robust_max-robust_min) * new_max

    return volume


def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. Can be an int, sequence or 1d numpy array of
    size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :return: cropped volume, and corresponding affine matrix if aff is not None.
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    vol_shape = volume.shape
    n_dims, _ = utils.get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = utils.reformat_to_list(cropping_margin, length=n_dims)
        min_crop_idx = cropping_margin
        max_crop_idx = [vol_shape[i] - cropping_margin[i] for i in range(n_dims)]
        assert (np.array(max_crop_idx) >= np.array(min_crop_idx)).all(), 'cropping_margin is larger than volume shape'
    else:
        cropping_shape = utils.reformat_to_list(cropping_shape, length=n_dims)
        min_crop_idx = [int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)]
        max_crop_idx = [min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)]
        assert (np.array(min_crop_idx) >= 0).all(), 'cropping_shape is larger than volume shape'
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        volume = volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        volume = volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        return volume, aff
    else:
        return volume


def crop_volume_around_region(volume, mask=None, threshold=0.1, masking_labels=None, margin=0, aff=None):
    """Crop a volume around a specific region. This region is defined by a mask obtained by either
    1) directly specifying it as input
    2) thresholding the input volume
    3) keeping a set of label values if the volume is a label map.
    :param volume: a 2d or 3d numpy array
    :param mask: (optional) mask of region to crop around. Must be same size as volume. Can either be boolean or 0/1.
    it defaults to masking around all values above threshold.
    :param threshold: (optional) if mask is None, lower bound to determine values to crop around
    :param masking_labels: (optional) if mask is None, and if the volume is a label map, it can be cropped around a
    set of labels specified in masking_labels, which can either be a single int, a sequence or a 1d numpy array.
    :param margin: (optional) add margin around mask
    :param aff: (optional) if specified, this function returns an updated affine matrix of the volume after cropping.
    :return: the cropped volume, the cropping indices (in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...]),
    and the updated affine matrix if aff is not None.
    """

    n_dims, _ = utils.get_dims(volume.shape)

    # mask ROIs for cropping
    if mask is None:
        if masking_labels is not None:
            masked_volume, mask = mask_label_map(volume, masking_values=masking_labels, return_mask=True)
        else:
            mask = volume > threshold

    # find cropping indices
    indices = np.nonzero(mask)
    min_idx = np.maximum(np.array([np.min(idx) for idx in indices]) - margin, 0)
    max_idx = np.minimum(np.array([np.max(idx) for idx in indices]) + 1 + margin, np.array(volume.shape))
    cropping = np.concatenate([min_idx, max_idx])

    # crop volume
    if n_dims == 3:
        volume = volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2], ...]
    elif n_dims == 2:
        volume = volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], ...]
    else:
        raise ValueError('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ min_idx
        return volume, cropping, aff
    else:
        return volume, cropping


def crop_volume_with_idx(volume, crop_idx, aff=None):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if specified, this function returns an updated affine matrix of the volume after cropping.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # crop image
    n_dims = int(crop_idx.shape[0] / 2)
    if n_dims == 2:
        volume = volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        volume = volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return volume, aff
    else:
        return volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """
    # get info
    vol_shape = volume.shape
    n_dims, _ = utils.get_dims(vol_shape)
    n_channels = len(vol_shape) - len(vol_shape[:n_dims])
    padding_shape = utils.reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # get padding margins
    min_pad_margins = np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)) / 2))
    max_pad_margins = np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)) / 2))
    if (min_pad_margins < 0).any():
        raise ValueError('volume is bigger than provided shape')
    pad_margins = tuple([(min_pad_margins[i], max_pad_margins[i]) for i in range(n_dims)])
    if n_channels > 1:
        pad_margins += [[0, 0]]

    # pad volume
    volume = np.pad(volume, pad_margins, mode='constant', constant_values=padding_value)

    if aff is not None:
        aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_pad_margins
        return volume, aff
    else:
        return volume


def flip_volume(volume, axis=None, direction=None, aff=None):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :return: flipped volume
    """

    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(volume, axis=axis)


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = utils.get_dims(volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = np.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = np.flip(volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)

    if return_aff:
        return volume, aff_flo
    else:
        return volume


def blur_volume(volume, sigma, mask=None):
    """Blur volume with gaussian masks of given sigma.
    :param volume: 2d or 3d numpy array
    :param sigma: standard deviation of the gaussian kernels. Can be a number, a sequence or a 1d numpy array
    :param mask: (optional) numpy array of the same shape as volume to correct for edge blurring effects.
    Mask can be a boolean or numerical array. In the later, the mask is computed by keeping all values above zero.
    :return: blurred volume
    """

    # initialisation
    n_dims, _ = utils.get_dims(volume.shape)
    sigma = utils.reformat_to_list(sigma, length=n_dims, dtype='float')

    # blur image
    volume = gaussian_filter(volume, sigma=sigma, mode='nearest')  # nearest refers to edge padding

    # correct edge effect if mask is not None
    if mask is not None:
        assert volume.shape == mask.shape, 'volume and mask should have the same dimensions: ' \
                                           'got {0} and {1}'.format(volume.shape, mask.shape)
        mask = (mask > 0) * 1.0
        blurred_mask = gaussian_filter(mask, sigma=sigma, mode='nearest')
        volume = volume / (blurred_mask + 1e-6)
        volume[mask == 0] = 0

    return volume


# --------------------------------------------------- edit label map ---------------------------------------------------

def correct_label_map(labels, list_incorrect_labels, list_correct_labels, smooth=False):
    """This function corrects specified label values in a label map by other given values.
    :param labels: a 2d or 3d label map
    :param list_incorrect_labels: list of all label values to correct (eg [1, 2, 3]). Can also be a path to such a list.
    :param list_correct_labels: list of correct label values.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list whithin
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param smooth: (optional) whether to smooth the corrected label map
    :return: corrected label map
    """

    # initialisation
    list_incorrect_labels = utils.load_array_if_path(list_incorrect_labels)
    list_correct_labels = utils.load_array_if_path(list_correct_labels)
    volume_labels = np.unique(labels)
    n_dims, _ = utils.get_dims(labels.shape)
    previous_correct_labels = None
    distance_map_list = None

    # loop over label values
    for incorrect_label, correct_label in zip(list_incorrect_labels, list_correct_labels):
        if incorrect_label in volume_labels:

            # only one possible value to replace with
            if isinstance(correct_label, (int, float, np.int64, np.int32, np.int16, np.int8)):
                incorrect_voxels = np.where(labels == incorrect_label)
                labels[incorrect_voxels] = correct_label

            # several possibilities
            elif isinstance(correct_label, (tuple, list)):
                mask = np.zeros(labels.shape, dtype='bool')

                # crop around label to correct
                for lab in correct_label:
                    mask = mask | (labels == lab)
                _, cropping = crop_volume_around_region(mask, margin=10)
                if n_dims == 2:
                    tmp_im = labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...]
                elif n_dims == 3:
                    tmp_im = labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...]
                else:
                    raise ValueError('cannot correct volumes with more than 3 dimensions')

                # calculate distance maps for all new label candidates
                incorrect_voxels = np.where(tmp_im == incorrect_label)
                if correct_label != previous_correct_labels:
                    distance_map_list = [distance_transform_edt(np.logical_not(tmp_im == lab))
                                         for lab in correct_label]
                    previous_correct_labels = correct_label
                distances_correct = np.stack([dist[incorrect_voxels] for dist in distance_map_list])

                # select nearest value
                idx_correct_lab = np.argmin(distances_correct, axis=0)
                tmp_im[incorrect_voxels] = np.array(correct_label)[idx_correct_lab]
                if n_dims == 2:
                    labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...] = tmp_im
                else:
                    labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...] = tmp_im

    # smoothing
    if smooth:
        kernel = np.ones(tuple([3] * n_dims))
        labels = smooth_label_map(labels, kernel)

    return labels


def mask_label_map(labels, masking_values, masking_value=0, return_mask=False):
    """
    This function masks a label map around a list of specified values.
    :param labels: input label map
    :param masking_values: list of values to mask around
    :param masking_value: (optional) value to mask the label map with
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked label map, and the applied mask if return_mask is True.
    """

    # build mask and mask labels
    mask = np.zeros(labels.shape, dtype=bool)
    masked_labels = labels.copy()
    for value in masking_values:
        mask = mask | (labels == value)
    masked_labels[np.logical_not(mask)] = masking_value

    if return_mask:
        mask = mask * 1
        return masked_labels, mask
    else:
        return masked_labels


def smooth_label_map(labels, kernel, print_progress=0):
    """This function smooth an input label map by replacing each voxel by the value of its most numerous neigbour.
    :param labels: input label map
    :param kernel: kernel when counting neighbours. Must contain only zeros or ones.
    :param print_progress: (optional) If not 0, interval at which to print the number of processed labels.
    :return: smoothed label map
    """
    # get info
    labels_shape = labels.shape
    label_list = np.unique(labels).astype('int32')

    # loop through label values
    count = np.zeros(labels_shape)
    labels_smoothed = np.zeros(labels_shape, dtype='int')
    for la, label in enumerate(label_list):
        if print_progress:
            utils.print_loop_info(la, len(label_list), print_progress)

        # count neigbours with same value
        mask = (labels == label) * 1
        n_neighbours = convolve(mask, kernel)

        # update label map and maximum neigbour counts
        idx = n_neighbours > count
        count[idx] = n_neighbours[idx]
        labels_smoothed[idx] = label
        labels_smoothed = labels_smoothed.astype('int32')

    return labels_smoothed


def erode_label_map(labels, labels_to_erode, erosion_factors=1., gpu=False, model=None, return_model=False):
    """Erode a given set of label values within a label map.
    :param labels: a 2d or 3d label map
    :param labels_to_erode: list of label values to erode
    :param erosion_factors: (optional) list of erosion factors to use for each label. If values are integers, normal
    erosion applies. If float, we first 1) blur a mask of the corresponding label value, and 2) use the erosion factor
    as a threshold in the blurred mask.
    If erosion_factors is a single value, the same factor will be applied to all labels.
    :param gpu: (optionnal) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param model: (optionnal) gpu model for blurring masks (if erosion factors are floats)
    :param return_model: (optional) whether to return the gpu blurring model
    :return: eroded label map, and gpu blurring model is return_model is True.
    """
    # reformat labels_to_erode and erode
    labels_to_erode = utils.reformat_to_list(labels_to_erode)
    erosion_factors = utils.reformat_to_list(erosion_factors, length=len(labels_to_erode))
    labels_shape = list(labels.shape)
    n_dims, _ = utils.get_dims(labels_shape)

    # loop over labels to erode
    for label_to_erode, erosion_factor in zip(labels_to_erode, erosion_factors):

        assert erosion_factor > 0, 'all erosion factors should be strictly positive, had {}'.format(erosion_factor)

        # get mask of current label value
        mask = (labels == label_to_erode)

        # erode as usual if erosion factor is int
        if int(erosion_factor) == erosion_factor:
            erode_struct = utils.build_binary_structure(int(erosion_factor), n_dims)
            eroded_mask = binary_erosion(mask, erode_struct)

        # blur mask and use erosion factor as a threshold if float
        else:
            if gpu:
                if model is None:
                    mask_in = KL.Input(shape=labels_shape + [1], dtype='float32')
                    list_k = get_gaussian_1d_kernels([1] * 3)
                    blurred_mask = blur_tensor(mask_in, list_k, n_dims=n_dims)
                    model = Model(inputs=mask_in, outputs=blurred_mask)
                eroded_mask = model.predict(utils.add_axis(np.float32(mask), -2))
            else:
                eroded_mask = blur_volume(np.array(mask, dtype='float32'), 1)
            eroded_mask = np.squeeze(eroded_mask) > erosion_factor

        # crop label map and mask around values to change
        mask = mask & np.logical_not(eroded_mask)
        cropped_lab_mask, cropping = crop_volume_around_region(mask, margin=3)
        croppped_labels = crop_volume_with_idx(labels, cropping)

        # calculate distance maps for all labels in cropped_labels
        labels_list = np.unique(croppped_labels)
        labels_list = labels_list[labels_list != label_to_erode]
        list_dist_maps = [distance_transform_edt(np.logical_not(croppped_labels == la)) for la in labels_list]
        candidate_distances = np.stack([dist[cropped_lab_mask] for dist in list_dist_maps])

        # select nearest value and put cropped labels back to full label map
        idx_correct_lab = np.argmin(candidate_distances, axis=0)
        croppped_labels[cropped_lab_mask] = np.array(labels_list)[idx_correct_lab]
        if n_dims == 2:
            labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...] = croppped_labels
        elif n_dims == 3:
            labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...] = croppped_labels

        if return_model:
            return labels, model
        else:
            return labels


def compute_hard_volumes(labels, voxel_volume=1., label_list=None, skip_background=True):
    """Compute hard volumes in a label map.
    :param labels: a label map
    :param voxel_volume: (optional) volume of voxel. Default is 1 (i.e. returned volumes are voxel counts).
    :param label_list: (optional) list of labels to compute volumes for. Can be an int, a sequence, or a numpy array.
    If None, the volumes of all label values are computed.
    :param skip_background: (optional) whether to skip computing the volume of the background.
    If label_list is None, this assumes background value is 0.
    If label_list is not None, this assumes the background is the first value in label list.
    :return: numpy 1d vector with the volumes of each structure
    """

    # initialisation
    subject_label_list = utils.reformat_to_list(np.unique(labels), dtype='int')
    if label_list is None:
        label_list = subject_label_list
    else:
        label_list = utils.reformat_to_list(label_list)
    if skip_background:
        label_list = label_list[1:]
    volumes = np.zeros(len(label_list))

    # loop over label values
    for idx, label in enumerate(label_list):
        if label in subject_label_list:
            mask = (labels == label) * 1
            volumes[idx] = np.sum(mask)
        else:
            volumes[idx] = 0

    return volumes * voxel_volume


def compute_distance_map(labels, masking_labels=None, crop_margin=None):
    """Compute distance map for a given list of label values in a label map.
    :param labels: a label map
    :param masking_labels: (optional) list of label values to mask the label map with. The distances will be computed
    for these labels only. default is None, where all positive values are considered.
    :return: a distance map with positive values inside the considered regions, and negative values outside."""

    n_dims, _ = utils.get_dims(labels.shape)

    # crop label map if necessary
    if crop_margin is not None:
        tmp_labels, crop_idx = crop_volume_around_region(labels, margin=crop_margin)
    else:
        tmp_labels = labels
        crop_idx = None

    # mask label map around specify values
    if masking_labels is not None:
        masking_labels = utils.reformat_to_list(masking_labels)
        mask = np.zeros(tmp_labels.shape, dtype='bool')
        for masking_label in masking_labels:
            mask = mask | tmp_labels == masking_label
    else:
        mask = tmp_labels > 0
    not_mask = np.logical_not(mask)

    # compute distances
    dist_in = distance_transform_edt(mask)
    dist_in = np.where(mask, dist_in - 0.5, dist_in)
    dist_out = - distance_transform_edt(not_mask)
    dist_out = np.where(not_mask, dist_out + 0.5, dist_out)
    tmp_dist = dist_in + dist_out

    # put back in original matrix if we cropped
    if crop_idx is not None:
        dist = np.min(tmp_dist) * np.ones(labels.shape, dtype='float32')
        if n_dims == 3:
            dist[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...] = tmp_dist
        elif n_dims == 2:
            dist[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...] = tmp_dist
    else:
        dist = tmp_dist

    return dist


# ------------------------------------------------- edit volumes in dir ------------------------------------------------

def mask_images_in_dir(image_dir, result_dir, mask_dir=None, threshold=0.1, dilate=0, erode=0, fill_holes=False,
                       masking_value=0, write_mask=False, mask_result_dir=None, recompute=True):
    """Mask all volumes in a folder, either with masks in a specified folder, or by keeping only the intensity values
    above a specified threshold.
    :param image_dir: path of directory with images to mask
    :param result_dir: path of directory where masked images will be writen
    :param mask_dir: (optional) path of directory containing masks. Masks are matched to images by sorting order.
    Mask volumes don't have to be boolean or 0/1 arrays as all strictly positive values are used to build the masks.
    Masks should have the same size as images. If images are multi-channel, masks can either be uni- or multi-channel.
    In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed masks.
    :param erode: (optional) number of voxels by which to erode the provided or computed masks.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed masks.
    :param masking_value: (optional) masking value
    :param write_mask: (optional) whether to write the applied masks
    :param mask_result_dir: (optional) path of resulting masks, if write_mask is True
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)
    if mask_result_dir is not None:
        utils.mkdir(mask_result_dir)

    # get path masks if necessary
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)

    # loop over images
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # mask images
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
            else:
                mask = None
            im = mask_volume(im, mask, threshold, dilate, erode, fill_holes, masking_value, write_mask)

            # write mask if necessary
            if write_mask:
                assert mask_result_dir is not None, 'if write_mask is True, mask_result_dir has to be specified as well'
                mask_result_path = os.path.join(mask_result_dir, os.path.basename(path_image))
                utils.save_volume(im[1], aff, h, mask_result_path)
                utils.save_volume(im[0], aff, h, path_result)
            else:
                utils.save_volume(im, aff, h, path_result)


def rescale_images_in_dir(image_dir, result_dir,
                          new_min=0, new_max=255,
                          min_percentile=0.025, max_percentile=0.975, use_positive_only=True,
                          recompute=True):
    """This function linearly rescales all volumes in image_dir between new_min and new_max.
    :param image_dir: path of directory with images to rescale
    :param result_dir: path of directory where rescaled images will be writen
    :param new_min: (optional) minimum value for the rescaled images.
    :param new_max: (optional) maximum value for the rescaled images.
    :param min_percentile: (optional) percentile for estimating robust minimum of each image.
    :param max_percentile: (optional) percentile for estimating robust maximum of each image.
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im = rescale_volume(im, new_min, new_max, min_percentile, max_percentile, use_positive_only)
            utils.save_volume(im, aff, h, path_result)


def crop_images_in_dir(image_dir, result_dir, cropping_margin=None, cropping_shape=None, recompute=True):
    """Crop all volumes in a folder by a given margin, or to a given shape.
    :param image_dir: path of directory with images to rescale
    :param result_dir: path of directory where cropped images will be writen
    :param cropping_margin: (optional) margin by which to crop the volume.
    Can be an int, a sequence or a 1d numpy array. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped.
    Can be an int, a sequence or a 1d numpy array. Should be given if cropping_margin is None.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # loop over images and masks
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # crop image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            volume, aff, h = utils.load_volume(path_image, im_only=False)
            volume, aff = crop_volume(volume, cropping_margin, cropping_shape, aff)
            utils.save_volume(volume, aff, h, path_result)


def crop_images_around_region_in_dir(image_dir,
                                     result_dir,
                                     mask_dir=None,
                                     threshold=0.1,
                                     masking_labels=None,
                                     crop_margin=5,
                                     recompute=True):
    """Crop all volumes in a folder around a region, which is defined for each volume by a mask obtained by either
    1) directly providing it as input
    2) thresholding the input volume
    3) keeping a set of label values if the volume is a label map.
    :param image_dir: path of directory with images to crop
    :param result_dir: path of directory where cropped images will be writen
    :param mask_dir: (optional) path of directory of input masks
    :param threshold: (optional) lower bound to determine values to crop around
    :param masking_labels: (optional) if the volume is a label map, it can be cropped around a given set of labels by
    specifying them in masking_labels, which can either be a single int, a list or a 1d numpy array.
    :param crop_margin: (optional) cropping margin
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # list volumes and masks
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)

    # loop over images and masks
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # crop image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            volume, aff, h = utils.load_volume(path_image, im_only=True)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
            else:
                mask = None
            volume, cropping, aff = crop_volume_around_region(volume, mask, threshold, masking_labels, crop_margin, aff)
            utils.save_volume(volume, aff, h, path_result)


def pad_images_in_dir(image_dir, result_dir, max_shape=None, padding_value=0, recompute=True):
    """Pads all the volumes in a folder to the same shape (either provided or computed).
    :param image_dir: path of directory with images to pad
    :param result_dir: path of directory where padded images will be writen
    :param max_shape: (optional) shape to pad the volumes to. Can be an int, a sequence or a 1d numpy array.
    If None, volumes will be padded to the shape of the biggest volume in image_dir.
    :param padding_value: (optional) value to pad the volumes with.
    :param recompute: (optional) whether to recompute result files even if they already exists
    :return: shape of the padded volumes.
    """

    # create result dir
    utils.mkdir(result_dir)

    # list labels
    path_images = utils.list_images_in_folder(image_dir)

    # get maximum shape
    if max_shape is None:
        max_shape, aff, _, _, h, _ = utils.get_volume_info(path_images[0])
        for path_image in path_images[1:]:
            image_shape, aff, _, _, h, _ = utils.get_volume_info(path_image)
            max_shape = tuple(np.maximum(np.asarray(max_shape), np.asarray(image_shape)))
        max_shape = np.array(max_shape)

    # loop over label maps
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 5)

        # pad map
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im, aff = pad_volume(im, max_shape, padding_value, aff)
            utils.save_volume(im, aff, h, path_result)

    return max_shape


def flip_images_in_dir(image_dir, result_dir, axis=None, direction=None, recompute=True):
    """Flip all images in a diretory along a specified axis.
    If unknown, this axis can be replaced by an anatomical direction.
    :param image_dir: path of directory with images to flip
    :param result_dir: path of directory where flipped images will be writen
    :param axis: (optional) axis along which to flip the volume
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' (anterior/posterior), 'si' (superior/inferior).
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    utils.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # flip image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im = flip_volume(im, axis=axis, direction=direction, aff=aff)
            utils.save_volume(im, aff, h, path_result)


def align_images_in_dir(image_dir, result_dir, aff_ref=None, path_ref_image=None, recompute=True):
    """This function aligns all images in image_dir to a reference orientation (axes and directions).
    This reference orientation can be directly provided as an affine matrix, or can be specified by a reference volume.
    If neither are provided, the reference orientation is assumed to be an identity matrix.
    :param image_dir: path of directory with images to align
    :param result_dir: path of directory where flipped images will be writen
    :param aff_ref: (optional) reference affine matrix. Can be a numpy array, or the path to such array.
    :param path_ref_image: (optional) path of a volume to which all images will be aligned.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # read reference affine matrix
    if path_ref_image is not None:
        _, aff_ref, _ = utils.load_volume(path_ref_image, im_only=False)
    elif aff_ref is not None:
        aff_ref = utils.load_array_if_path(aff_ref)
    else:
        aff_ref = np.eye(4)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # align image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im, aff = align_volume_to_ref(im, aff, aff_ref=aff_ref, return_aff=True)
            utils.save_volume(im, aff_ref, h, path_result)


def correct_nans_images_in_dir(image_dir, result_dir, recompute=True):
    """Correct NaNs in all images in a directory.
    :param image_dir: path of directory with images to correct
    :param result_dir: path of directory where corrected images will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    utils.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # flip image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im[np.isnan(im)] = 0
            utils.save_volume(im, aff, h, path_result)


def blur_images_in_dir(image_dir, result_dir, sigma, mask_dir=None, gpu=False, recompute=True):
    """This function blurs all the images in image_dir with kernels of the specified std deviations.
    :param image_dir: path of directory with images to blur
    :param result_dir: path of directory where blurred images will be writen
    :param sigma: standard deviation of the blurring gaussian kernels.
    Can be a number (isotropic blurring), or a sequence witht the same length as the number of dimensions of images.
    :param mask_dir: (optional) path of directory with masks of the region to blur.
    Images and masks are matched by sorting order.
    :param gpu: (optional) whether to use a fast gpu model for blurring
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # list images and masks
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)

    # loop over images
    previous_model_input_shape = None
    model = None
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, im_shape, aff, n_dims, _, h, image_res = utils.get_volume_info(path_image, return_volume=True)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
                assert mask.shape == im.shape, 'mask and image should have the same shape'
            else:
                mask = None

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    image_in = [KL.Input(shape=im_shape + [1])]
                    sigma = utils.reformat_to_list(sigma, length=n_dims)
                    kernels_list = get_gaussian_1d_kernels(sigma)
                    image = blur_tensor(image_in[0], kernels_list, n_dims)
                    if mask is not None:
                        image_in.append(KL.Input(shape=im_shape + [1], dtype='float32'))  # mask
                        masked_mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                                                   tf.zeros_like(x, dtype='float32')))(image_in[1])
                        blurred_mask = blur_tensor(masked_mask, kernels_list, n_dims)
                        image = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([image, blurred_mask])
                        image = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0],
                                                             tf.zeros_like(x[0])))([image, masked_mask])
                    model = Model(inputs=image_in, outputs=image)
                if mask is None:
                    im = np.squeeze(model.predict(utils.add_axis(im, -2)))
                else:
                    im = np.squeeze(model.predict([utils.add_axis(im, -2), utils.add_axis(mask, -2)]))
            else:
                im = blur_volume(im, sigma, mask=mask)
            utils.save_volume(im, aff, h, path_result)


def create_mutlimodal_images(list_channel_dir, result_dir, recompute=True):
    """This function forms multimodal images by stacking channels located in different folders.
    :param list_channel_dir: list of all directories, each containing the same channel for allimages.
    Channels are matched between folders by sorting order.
    :param result_dir: path of directory where multimodal images will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    if not isinstance(list_channel_dir, list):
        raise TypeError('list_channel_dir should be a list')

    # gather path of all images for all channels
    list_channel_paths = [utils.list_images_in_folder(d) for d in list_channel_dir]
    n_images = len(list_channel_paths[0])
    n_channels = len(list_channel_dir)
    for channel_paths in list_channel_paths:
        if len(channel_paths) != n_images:
            raise ValueError('all directories should have the same number of files')

    # loop over images
    for idx in range(n_images):
        utils.print_loop_info(idx, n_images, 10)

        # stack all channels and save multichannel image
        path_result = os.path.join(result_dir, os.path.basename(list_channel_paths[0][idx]))
        if (not os.path.isfile(path_result)) | recompute:
            list_channels = list()
            tmp_aff = None
            tmp_h = None
            for channel_idx in range(n_channels):
                tmp_channel, tmp_aff, tmp_h = utils.load_volume(list_channel_paths[channel_idx][idx], im_only=False)
                list_channels.append(tmp_channel)
            im = np.stack(list_channels, axis=-1)
            utils.save_volume(im, tmp_aff, tmp_h, path_result)


def convert_images_in_dir_to_nifty(image_dir, result_dir, aff=None, recompute=True):
    """Converts all images in image_dir to nifty format.
    :param image_dir: path of directory with images to convert
    :param result_dir: path of directory where converted images will be writen
    :param aff: (optional) affine matrix in homogeneous coordinates with which to write the images.
    Can also be 'FS' to write images with FreeSurfer typical affine matrix.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # convert images to nifty format
        path_result = os.path.join(result_dir, os.path.basename(utils.strip_extension(path_image))) + '.nii.gz'
        if (not os.path.isfile(path_result)) | recompute:
            im, tmp_aff, h = utils.load_volume(path_image, im_only=False)
            if aff is not None:
                tmp_aff = aff
            utils.save_volume(im, tmp_aff, h, path_result)


def mri_convert_images_in_dir(image_dir,
                              result_dir,
                              interpolation=None,
                              reference_dir=None,
                              same_reference=False,
                              voxsize=None,
                              path_freesurfer='/usr/local/freesurfer',
                              mri_convert_path='/usr/local/freesurfer/bin/mri_convert',
                              recompute=True):
    """This function launches mri_convert on all images contained in image_dir, and writes the results in result_dir.
    The interpolation type can be specified (i.e. 'nearest'), as well as a folder containing references for resampling.
    reference_dir can be the path of a single *image* if same_reference=True.
    :param image_dir: path of directory with images to convert
    :param result_dir: path of directory where converted images will be writen
    :param interpolation: (optional) interpolation type, can be 'inter' (default), 'cubic', 'nearest', 'trilinear'
    :param reference_dir: (optional) path of directory with reference images. References are matched to images by
    sorting order. If same_reference is false, references and images are matched by sorting order.
    This can also be the path to a single image that will be used as reference for all images im image_dir (set
    same_reference to true in that case).
    :param same_reference: (optional) whether to use a single image as reference for all images to interpolate.
    :param voxsize: (optional) resolution at which to resample converted image. Must be a list of length n_dims.
    :param path_freesurfer: (optional) path FreeSurfer home
    :param mri_convert_path: (optional) path mri_convert binary file
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = mri_convert_path + ' '

    # list images
    path_images = utils.list_images_in_folder(image_dir)
    if reference_dir is not None:
        if same_reference:
            path_references = [reference_dir] * len(path_images)
        else:
            path_references = utils.list_images_in_folder(reference_dir)
            assert len(path_references) == len(path_images), 'different number of files in image_dir and reference_dir'
    else:
        path_references = [None] * len(path_images)

    # loop over images
    for idx, (path_image, path_reference) in enumerate(zip(path_images, path_references)):
        utils.print_loop_info(idx, len(path_images), 10)

        # convert image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            cmd = mri_convert + path_image + ' ' + path_result + ' -odt float'
            if interpolation is not None:
                cmd += ' -rt ' + interpolation
            if reference_dir is not None:
                cmd += ' -rl ' + path_reference
            if voxsize is not None:
                voxsize = utils.reformat_to_list(voxsize, dtype='float')
                cmd += ' --voxsize ' + ' '.join([str(np.around(v, 3)) for v in voxsize])
            os.system(cmd)


def samseg_images_in_dir(image_dir,
                         result_dir,
                         atlas_dir=None,
                         threads=4,
                         path_freesurfer='/usr/local/freesurfer',
                         keep_segm_only=True,
                         recompute=True):
    """This function launches samseg for all images contained in image_dir and writes the results in result_dir.
    If keep_segm_only=True, the result segmentation is copied in result_dir and SAMSEG's intermediate result dir is
    deleted.
    :param image_dir: path of directory with input images
    :param result_dir: path of directory where processed images folders (if keep_segm_only is False),
    or samseg segmentation (if keep_segm_only is True) will be writen
    :param atlas_dir: (optional) path of samseg atlas directory. If None, use samseg default atlas.
    :param threads: (optional) number of threads to use
    :param path_freesurfer: (optional) path FreeSurfer home
    :param keep_segm_only: (optional) whether to keep samseg result folders, or only samseg segmentations.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    path_samseg = os.path.join(path_freesurfer, 'bin', 'run_samseg')

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # build path_result
        path_im_result_dir = os.path.join(result_dir, utils.strip_extension(os.path.basename(path_image)))
        path_samseg_result = os.path.join(path_im_result_dir, ''.join(
            os.path.basename(path_image).split('.')[:-1]) + '_crispSegmentation.nii')
        if keep_segm_only:
            path_result = os.path.join(result_dir, os.path.basename(path_image))
        else:
            path_result = path_samseg_result

        # run samseg
        if (not os.path.isfile(path_result)) | recompute:
            cmd = path_samseg + ' -i ' + path_image + ' -o ' + path_im_result_dir + ' --threads ' + str(threads)
            if atlas_dir is not None:
                cmd += ' --a ' + atlas_dir
            os.system(cmd)

        # move segmentation to result_dir if necessary
        if keep_segm_only:
            if os.path.isfile(path_samseg_result):
                shutil.move(path_samseg_result, path_result)
            if os.path.isdir(path_im_result_dir):
                shutil.rmtree(path_im_result_dir)


def simulate_upsampled_anisotropic_images(image_dir,
                                          downsample_image_result_dir,
                                          resample_image_result_dir,
                                          data_res,
                                          labels_dir=None,
                                          downsample_labels_result_dir=None,
                                          slice_thickness=None,
                                          path_freesurfer='/usr/local/freesurfer/',
                                          gpu=False,
                                          recompute=True):
    """This function takes as input a set of HR images and creates two datasets with it:
    1) a set of LR images obtained by downsampling the HR images with nearest neighbour interpolation,
    2) a set of HR images obtained by resampling the LR images to native HR with linear interpolation.
    Additionally, this function can also create a set of LR labels from label maps corresponding to the input images.
    :param image_dir: path of directory with input images (only uni-model images supported)
    :param downsample_image_result_dir: path of directory where downsampled images will be writen
    :param resample_image_result_dir: path of directory where resampled images will be writen
    :param data_res: resolution of LR images. Can either be: an int, a float, a list or a numpy array.
    :param labels_dir: (optional) path of directory with label maps corresponding to input images
    :param downsample_labels_result_dir: (optional) path of directory where downsampled label maps will be writen
    :param slice_thickness: (optional) thickness of slices to simulate. Can be a number, a list or a numpy array.
    :param path_freesurfer: (optional) path freesurfer home, as this function uses mri_convert
    :param gpu: (optional) whether to use a fast gpu model for blurring
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    utils.mkdir(resample_image_result_dir)
    utils.mkdir(downsample_image_result_dir)
    if labels_dir is not None:
        assert downsample_labels_result_dir is not None, \
            'downsample_labels_result_dir should not be None if labels_dir is specified'
        utils.mkdir(downsample_labels_result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin') + ' '

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    if labels_dir is not None:
        path_labels = utils.list_images_in_folder(labels_dir)
    else:
        path_labels = [None] * len(path_images)

    # initialisation
    _, _, n_dims, _, _, image_res = utils.get_volume_info(path_images[0], return_volume=False)
    data_res = np.squeeze(utils.reformat_to_n_channels_array(data_res, n_dims, n_channels=1))
    slice_thickness = utils.reformat_to_list(slice_thickness, length=n_dims)

    # loop over images
    previous_model_input_shape = None
    model = None
    for idx, (path_image, path_labels) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # downsample image
        path_im_downsampled = os.path.join(downsample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_downsampled)) | recompute:
            im, im_shape, aff, n_dims, _, h, image_res = utils.get_volume_info(path_image, return_volume=True)
            sigma = utils.get_std_blurring_mask_for_downsampling(data_res, image_res, thickness=slice_thickness)

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    image_in = KL.Input(shape=im_shape + [1])
                    kernels_list = get_gaussian_1d_kernels(sigma)
                    kernels_list = [None if data_res[i] == image_res[i] else kernels_list[i] for i in range(n_dims)]
                    image = blur_tensor(image_in, kernels_list, n_dims)
                    model = Model(inputs=image_in, outputs=image)
                im = np.squeeze(model.predict(utils.add_axis(im, -2)))
            else:
                im = blur_volume(im, sigma, mask=None)
            utils.save_volume(im, aff, h, path_im_downsampled)

            # downsample blurred image
            voxsize = ' '.join([str(r) for r in data_res])
            cmd = mri_convert + path_im_downsampled + ' ' + path_im_downsampled + ' --voxsize ' + voxsize
            cmd += ' -odt float -rt nearest'
            os.system(cmd)

        # downsample labels if necessary
        if path_labels is not None:
            path_lab_downsampled = os.path.join(downsample_labels_result_dir, os.path.basename(path_labels))
            if (not os.path.isfile(path_lab_downsampled)) | recompute:
                voxsize = ' '.join([str(r) for r in data_res])
                cmd = mri_convert + path_labels + ' ' + path_lab_downsampled + ' --voxsize ' + voxsize
                cmd += ' -odt float -rt nearest'
                os.system(cmd)

        # upsample image
        path_im_upsampled = os.path.join(resample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_upsampled)) | recompute:
            cmd = mri_convert + path_im_downsampled + ' ' + path_im_upsampled + ' -rl ' + path_image + ' -odt float'
            os.system(cmd)


def check_images_in_dir(image_dir, check_values=False, keep_unique=True):
    """Check if all volumes within the same folder share the same characteristics: shape, affine matrix, resolution.
    Also have option to check if all volumes have the same intensity values (useful for label maps).
    :return four lists, each containing the different values detected for a specific parameter among those to check."""

    # define information to check
    list_shape = list()
    list_aff = list()
    list_res = list()
    if check_values:
        list_unique_values = list()
    else:
        list_unique_values = None

    # loop through files
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # get info
        im, im_shape, aff, _, _, h, data_res = utils.get_volume_info(path_image, return_volume=True)
        aff = np.round(aff[:3, :3], 2).tolist()
        data_res = np.round(np.array(data_res), 2).tolist()

        # add values to list if not already there
        if (im_shape not in list_shape) | (not keep_unique):
            list_shape.append(im_shape)
        if (aff not in list_aff) | (not keep_unique):
            list_aff.append(aff)
        if (data_res not in list_res) | (not keep_unique):
            list_res.append(data_res)
        if list_unique_values is not None:
            uni = np.unique(im).tolist()
            if (uni not in list_unique_values) | (not keep_unique):
                list_unique_values.append(uni)

    return list_shape, list_aff, list_res, list_unique_values


# ----------------------------------------------- edit label maps in dir -----------------------------------------------

def correct_labels_in_dir(labels_dir, list_incorrect_labels, list_correct_labels, results_dir, smooth=False,
                          recompute=True):
    """This function corrects label values for all labels in a folder.
    :param labels_dir: path of directory with input label maps
    :param list_incorrect_labels: list of all label values to correct (e.g. [1, 2, 3, 4]).
    :param list_correct_labels: list of correct label values.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list whithin
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param results_dir: path of directory where corrected label maps will be writen
    :param smooth: (optional) whether to smooth the corrected label maps
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(results_dir)

    # prepare data files
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # correct labels
        path_result = os.path.join(results_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_label, im_only=False, dtype='int32')
            im = correct_label_map(im, list_incorrect_labels, list_correct_labels, smooth=smooth)
            utils.save_volume(im, aff, h, path_result)


def mask_labels_in_dir(labels_dir, result_dir, values_to_keep, masking_value=0, mask_result_dir=None, recompute=True):
    """This function masks all label maps in a folder by keeping a set of given label values.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where corrected label maps will be writen
    :param values_to_keep: list of values for masking the label maps.
    :param masking_value: (optional) value to mask the label maps with
    :param mask_result_dir: (optional) path of directory where applied masks will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)
    if mask_result_dir is not None:
        utils.mkdir(mask_result_dir)

    # reformat values to keep
    if isinstance(values_to_keep, (int, float)):
        values_to_keep = [values_to_keep]
    elif not isinstance(values_to_keep, (tuple, list)):
        raise TypeError('values to keep should be int, float, tuple, or list')

    # loop over labels
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # mask labels
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if mask_result_dir is not None:
            path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
        else:
            path_result_mask = ''
        if (not os.path.isfile(path_result)) | \
                (mask_result_dir is not None) & (not os.path.isfile(path_result_mask)) | \
                recompute:
            lab, aff, h = utils.load_volume(path_label, im_only=False)
            if mask_result_dir is not None:
                labels, mask = mask_label_map(lab, values_to_keep, masking_value, return_mask=True)
                path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
                utils.save_volume(mask, aff, h, path_result_mask)
            else:
                labels = mask_label_map(lab, values_to_keep, masking_value, return_mask=False)
            utils.save_volume(labels, aff, h, path_result)


def smooth_labels_in_dir(labels_dir, result_dir, gpu=False, path_label_list=None, recompute=True):
    """Smooth all label maps in a folder by replacing each voxel by the value of its most numerous neigbours.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where smoothed label maps will be writen
    :param gpu: (optional) whether to use a gpu implementation for faster processing
    :param path_label_list: (optionnal) if gpu is True, path of numpy array with all label values.
    Automatically computed if not provided.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)

    if gpu:
        # initialisation
        label_list, _ = utils.get_list_labels(label_list=path_label_list, labels_dir=labels_dir, FS_sort=True)
        previous_model_input_shape = None
        smoothing_model = None

        # loop over label maps
        for idx, path_label in enumerate(path_labels):
            utils.print_loop_info(idx, len(path_labels), 10)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path_label))
            if (not os.path.isfile(path_result)) | recompute:
                labels, label_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_label, return_volume=True)
                if label_shape != previous_model_input_shape:
                    previous_model_input_shape = label_shape
                    smoothing_model = smoothing_gpu_model(label_shape, label_list)
                labels = smoothing_model.predict(utils.add_axis(labels))
                utils.save_volume(np.squeeze(labels), aff, h, path_result, dtype='int')

    else:
        # build kernel
        _, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])
        kernel = np.ones(tuple([3] * n_dims))

        # loop over label maps
        for idx, path in enumerate(path_labels):
            utils.print_loop_info(idx, len(path_labels), 10)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path))
            if (not os.path.isfile(path_result)) | recompute:
                volume, aff, h = utils.load_volume(path, im_only=False)
                new_volume = smooth_label_map(volume, kernel)
                utils.save_volume(new_volume, aff, h, path_result, dtype='int')


def smoothing_gpu_model(label_shape, label_list):
    """This function builds a gpu model in keras with a tensorflow backend to smooth label maps.
    This model replaces each voxel of the input by the value of its most numerous neigbour.
    :param label_shape: shape of the label map
    :param label_list: list of all labels to consider
    :return: gpu smoothing model
    """

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_labels = label_list.shape[0]
    _, lut = utils.rearrange_label_list(label_list)

    # convert labels to new_label_list and use one hot encoding
    labels_in = KL.Input(shape=label_shape, name='lab_input', dtype='int32')
    labels = convert_labels(labels_in, lut)
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels)

    # count neighbouring voxels
    n_dims, _ = utils.get_dims(label_shape)
    kernel = KL.Lambda(lambda x: tf.convert_to_tensor(
        utils.add_axis(utils.add_axis(np.ones(tuple([n_dims] * n_dims)).astype('float32'), -1), -1)))([])
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_labels, axis=-1))(labels)
    labels = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[0], kernel])
    for i in range(1, n_labels):
        tmp = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[i], kernel])
        labels = KL.Lambda(lambda x: tf.concat([x[0], x[1]], -1))([labels, tmp])

    # take the argmax and convert labels to original values
    labels = KL.Lambda(lambda x: tf.math.argmax(x, -1))(labels)
    labels = convert_labels(labels, label_list)
    return Model(inputs=labels_in, outputs=labels)


def erode_labels_in_dir(labels_dir, result_dir, labels_to_erode, erosion_factors=1., gpu=False, recompute=True):
    """Erode a given set of label values for all label maps in a folder.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where cropped label maps will be writen
    :param labels_to_erode: list of label values to erode
    :param erosion_factors: (optional) list of erosion factors to use for each label value. If values are integers,
    normal erosion applies. If float, we first 1) blur a mask of the corresponding label value with a gpu model,
    and 2) use the erosion factor as a threshold in the blurred mask.
    If erosion_factors is a single value, the same factor will be applied to all labels.
    :param gpu: (optionnal) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    utils.mkdir(result_dir)

    # loop over label maps
    model = None
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 5)

        # erode label map
        labels, aff, h = utils.load_volume(path_label, im_only=False)
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            labels, model = erode_label_map(labels, labels_to_erode, erosion_factors, gpu, model, return_model=True)
            utils.save_volume(labels, aff, h, path_result)


def upsample_labels_in_dir(labels_dir,
                           target_res,
                           result_dir,
                           path_label_list=None,
                           path_freesurfer='/usr/local/freesurfer/',
                           recompute=True):
    """This funtion upsamples all label maps within a folder. Importantly, each label map is converted into probability
    maps for all label values, and all these maps are upsampled separetely. The upsampled label maps are recovered by
    taking the argmax of the label values probability maps.
    :param labels_dir: path of directory with label maps to upsample
    :param target_res: resolution at which to upsample the label maps. can be a single number (isotropic), or a list.
    :param result_dir: path of directory where the upsampled label maps will be writen
    :param path_label_list: (optional) path of numpy array containing all label values.
    Computed automatically if not given.
    :param path_freesurfer: (optional) path freesurfer home (upsampling performed with mri_convert)
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # prepare result dir
    utils.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin')

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)
    labels_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_labels[0])

    # build command
    target_res = utils.reformat_to_list(target_res, length=n_dims)
    post_cmd = ' -voxsize ' + ' '.join([str(r) for r in target_res]) + ' -odt float'

    # load label list and corresponding LUT to make sure that labels go from 0 to N-1
    label_list, _ = utils.get_list_labels(path_label_list, labels_dir=path_labels, FS_sort=True)
    new_label_list, lut = utils.rearrange_label_list(label_list)

    # loop over label maps
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 5)
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:

            # load volume
            labels, aff, h = utils.load_volume(path_label, im_only=False)
            labels = lut[labels.astype('int')]

            # create individual folders for label map
            basefilename = utils.strip_extension(os.path.basename(path_label))
            indiv_label_dir = os.path.join(result_dir, basefilename)
            upsample_indiv_label_dir = os.path.join(result_dir, basefilename + '_upsampled')
            utils.mkdir(indiv_label_dir)
            utils.mkdir(upsample_indiv_label_dir)

            # loop over label values
            for label in new_label_list:
                path_mask = os.path.join(indiv_label_dir, str(label)+'.nii.gz')
                path_mask_upsampled = os.path.join(upsample_indiv_label_dir, str(label)+'.nii.gz')
                if not os.path.isfile(path_mask):
                    mask = (labels == label) * 1.0
                    utils.save_volume(mask, aff, h, path_mask)
                if not os.path.isfile(path_mask_upsampled):
                    cmd = mri_convert + ' ' + path_mask + ' ' + path_mask_upsampled + post_cmd
                    os.system(cmd)

            # compute argmax of upsampled probability maps (upload them one at a time)
            probmax, aff, h = utils.load_volume(os.path.join(upsample_indiv_label_dir, '0.nii.gz'), im_only=False)
            labels = np.zeros(probmax.shape, dtype='int')
            for label in new_label_list:
                prob = utils.load_volume(os.path.join(upsample_indiv_label_dir, str(label) + '.nii.gz'))
                idx = prob > probmax
                labels[idx] = label
                probmax[idx] = prob[idx]
            utils.save_volume(label_list[labels], aff, h, path_result)


def compute_hard_volumes_in_dir(labels_dir,
                                voxel_volume=None,
                                path_label_list=None,
                                skip_background=True,
                                path_numpy_result=None,
                                path_csv_result=None,
                                FS_sort=False):
    """Compute hard volumes of structures for all label maps in a folder.
    :param labels_dir: path of directory with input label maps
    :param voxel_volume: (optional) volume of the voxels. If None, it will be directly inferred from the file header.
    Set to 1 for a voxel count.
    :param path_label_list: (optional) list of labels to compute volumes for.
    Can be an int, a sequence, or a numpy array. If None, the volumes of all label values are computed for each subject.
    :param skip_background: (optional) whether to skip computing the volume of the background.
    If label_list is None, this assumes background value is 0.
    If label_list is not None, this assumes the background is the first value in label list.
    :param path_numpy_result: (optional) path where to write the result volumes as a numpy array.
    :param path_csv_result: (optional) path where to write the results as csv file.
    :param FS_sort: (optional) whether to sort the labels in FreeSurfer order.
    :return: numpy array with the volume of each structure for all subjects.
    Rows represent label values, and columns represent subjects.
    """

    # create result directories
    if path_numpy_result is not None:
        utils.mkdir(os.path.dirname(path_numpy_result))
    if path_csv_result is not None:
        utils.mkdir(os.path.dirname(path_csv_result))

    # load or compute labels list
    label_list = utils.get_list_labels(path_label_list, labels_dir, FS_sort=FS_sort)

    # create csv volume file if necessary
    if path_csv_result is not None:
        if skip_background:
            cvs_header = [['subject'] + [str(lab) for lab in label_list[1:]]]
        else:
            cvs_header = [['subject'] + [str(lab) for lab in label_list]]
        with open(path_csv_result, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(cvs_header)
        csvFile.close()

    # loop over label maps
    path_labels = utils.list_images_in_folder(labels_dir)
    if skip_background:
        volumes = np.zeros((label_list.shape[0]-1, len(path_labels)))
    else:
        volumes = np.zeros((label_list.shape[0], len(path_labels)))
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # load segmentation, and compute unique labels
        labels, _, _, _, _, _, subject_res = utils.get_volume_info(path_label, return_volume=True)
        if voxel_volume is None:
            voxel_volume = float(np.prod(subject_res))
        subject_volumes = compute_hard_volumes(labels, voxel_volume, label_list, skip_background)
        volumes[:, idx] = subject_volumes

        # write volumes
        if path_csv_result is not None:
            subject_volumes = np.around(volumes[:, idx], 3)
            row = [utils.strip_suffix(os.path.basename(path_label))] + [str(vol) for vol in subject_volumes]
            with open(path_csv_result, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

    # write numpy array if necessary
    if path_numpy_result is not None:
        np.save(path_numpy_result, volumes)

    return volumes


def build_atlas(labels_dir, align_centre_of_mass=False, margin=15, path_result_atlas=None):
    """This function builds a binary atlas (defined by label values > 0) from several label maps.
    :param labels_dir: path of directory with input label maps
    :param align_centre_of_mass: whether to build the atlas by aligning the center of mass of each label map.
    If False, the atlas has the same size as the input label maps, which are assumed to be aligned.
    :param margin: (optional) If align_centre_of_mass is True, margin by which to crop the input label maps around
    their center of mass. Therefore it controls the size of the output atlas: (2*margin + 1)**n_dims.
    :param path_result_atlas: (optional) path where the output atlas will be writen.
    Default is None, where the atlas is not saved."""

    # list of all label maps
    path_labels = utils.list_images_in_folder(labels_dir)

    # create empty atlas
    if align_centre_of_mass:
        atlas = np.zeros([margin * 2] * 3)
    else:
        atlas = np.zeros(utils.load_volume(path_labels[0]).shape)

    # loop over label maps
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # load label map and build mask
        lab = (utils.load_volume(path_label, dtype='int32', aff_ref=np.eye(4)) > 0) * 1

        if align_centre_of_mass:
            # find centre of mass
            indices = np.where(lab > 0)
            centre_of_mass = np.array([np.mean(indices[0]), np.mean(indices[1]), np.mean(indices[2])], dtype='int32')
            # crop label map around centre of mass
            min_crop = centre_of_mass - margin
            max_crop = centre_of_mass + margin
            atlas += lab[min_crop[0]:max_crop[0], min_crop[1]:max_crop[1], min_crop[2]:max_crop[2]]
        else:
            atlas += lab

    # normalise atlas and save it if necessary
    atlas /= len(path_labels)
    if path_result_atlas is not None:
        utils.save_volume(atlas, None, None, path_result_atlas)

    return atlas


# ---------------------------------------------------- edit dataset ----------------------------------------------------

def check_images_and_labels(image_dir, labels_dir):
    """Check if corresponding images and labels have the same affine matrices and shapes.
    Labels are matched to images by sorting order.
    :param image_dir: path of directory with input images
    :param labels_dir: path of directory with corresponding label maps
    """

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), 'different number of files in image_dir and labels_dir'

    # loop over images and labels
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load images and labels
        im, aff_im, h_im = utils.load_volume(path_image, im_only=False)
        lab, aff_lab, h_lab = utils.load_volume(path_label, im_only=False)
        aff_im_list = np.round(aff_im, 2).tolist()
        aff_lab_list = np.round(aff_lab, 2).tolist()

        # check matching affine and shape
        if aff_lab_list != aff_im_list:
            print('aff mismatch :\n' + path_image)
            print(aff_im_list)
            print(path_label)
            print(aff_lab_list)
            print('')
        if lab.shape != im.shape:
            print('shape mismatch :\n' + path_image)
            print(im.shape)
            print('\n' + path_label)
            print(lab.shape)
            print('')


def crop_dataset_to_minimum_size(labels_dir,
                                 result_dir,
                                 image_dir=None,
                                 image_result_dir=None,
                                 margin=5):
    """Crop all label maps in a directory to the minimum possible common size, with a margin.
    This is achieved by cropping each label map individually to the minimum size, and by padding all the cropped maps to
    the same size (taken to be the maximum size of hte cropped maps).
    If images are provided, they undergo the same transformations as their corresponding label maps.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where cropped label maps will be writen
    :param image_dir: (optional) if not None, the cropping will be applied to all images in this directory
    :param image_result_dir: (optional) path of directory where cropped images will be writen
    :param margin: (optional) margin to apply around the label maps during cropping
    """

    # create result dir
    utils.mkdir(result_dir)
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        utils.mkdir(image_result_dir)

    # list labels and images
    path_labels = utils.list_images_in_folder(labels_dir)
    if image_dir is not None:
        path_images = utils.list_images_in_folder(image_dir)
    else:
        path_images = [None] * len(path_labels)
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])

    # loop over label maps for cropping
    print('\ncropping labels to individual minimum size')
    maximum_size = np.zeros(n_dims)
    for idx, (path_label, path_image) in enumerate(zip(path_labels, path_images)):
        utils.print_loop_info(idx, len(path_labels), 10)

        # crop label maps and update maximum size of cropped map
        label, aff, h = utils.load_volume(path_label, im_only=False)
        label, cropping, aff = crop_volume_around_region(label, aff=aff)
        utils.save_volume(label, aff, h, os.path.join(result_dir, os.path.basename(path_label)))
        maximum_size = np.maximum(maximum_size, np.array(label.shape) + margin*2)  # *2 to add margin on each side

        # crop images if required
        if path_image is not None:
            image, aff_im, h_im = utils.load_volume(path_image, im_only=False)
            image, aff_im = crop_volume_with_idx(image, cropping, aff=aff_im)
            utils.save_volume(image, aff_im, h_im, os.path.join(image_result_dir, os.path.basename(path_image)))

    # loop over label maps for padding
    print('\npadding labels to same size')
    for idx, (path_label, path_image) in enumerate(zip(path_labels, path_images)):
        utils.print_loop_info(idx, len(path_labels), 10)

        # pad label maps to maximum size
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        label, aff, h = utils.load_volume(path_result, im_only=False)
        label, aff = pad_volume(label, maximum_size, aff=aff)
        utils.save_volume(label, aff, h, path_result)

        # crop images if required
        if path_image is not None:
            path_result = os.path.join(image_result_dir, os.path.basename(path_image))
            image, aff, h = utils.load_volume(path_result, im_only=False)
            image, aff = pad_volume(image, maximum_size, aff=aff)
            utils.save_volume(image, aff, h, path_result)


def subdivide_dataset_to_patches(patch_shape,
                                 image_dir=None,
                                 image_result_dir=None,
                                 labels_dir=None,
                                 labels_result_dir=None,
                                 full_background=True,
                                 remove_after_dividing=False):
    """This function subdivides images and/or label maps into several smaller patches of specified shape.
    :param patch_shape: shape of patches to create. Can either be an int, a sequence, or a 1d numpy array.
    :param image_dir: (optional) path of directory with input images
    :param image_result_dir: (optional) path of directory where image patches will be writen
    :param labels_dir: (optional) path of directory with input label maps
    :param labels_result_dir: (optional) path of directory where label map patches will be writen
    :param full_background: (optional) whether to keep patches only labelled as background (only if label maps are
    provided).
    :param remove_after_dividing: (optional) whether to delete input images after having divided them in smaller
    patches. This enables to save disk space in the subdivision process.
    """

    # create result dir and list images and label maps
    assert (image_dir is not None) | (labels_dir is not None), \
        'at least one of image_dir or labels_dir should not be None.'
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        utils.mkdir(image_result_dir)
        path_images = utils.list_images_in_folder(image_dir)
    else:
        path_images = None
    if labels_dir is not None:
        assert labels_result_dir is not None, 'labels_result_dir should not be None if labels_dir is specified'
        utils.mkdir(labels_result_dir)
        path_labels = utils.list_images_in_folder(labels_dir)
    else:
        path_labels = None
    if path_images is None:
        path_images = [None] * len(path_labels)
    if path_labels is None:
        path_labels = [None] * len(path_images)

    # reformat path_shape
    patch_shape = utils.reformat_to_list(patch_shape)
    n_dims, _ = utils.get_dims(patch_shape)

    # loop over images and labels
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load image and labels
        if path_image is not None:
            im, aff_im, h_im = utils.load_volume(path_image, im_only=False, squeeze=False)
        else:
            im = aff_im = h_im = None
        if path_label is not None:
            lab, aff_lab, h_lab = utils.load_volume(path_label, im_only=False, squeeze=True)
        else:
            lab = aff_lab = h_lab = None

        # get volume shape
        if path_image is not None:
            shape = im.shape
        else:
            shape = lab.shape

        # crop image and label map to size divisible by patch_shape
        new_size = np.array([utils.find_closest_number_divisible_by_m(shape[i], patch_shape[i]) for i in range(n_dims)])
        crop = np.round((np.array(shape[:n_dims]) - new_size) / 2).astype('int')
        crop = np.concatenate((crop, crop + new_size), axis=0)
        if (im is not None) & (n_dims == 2):
            im = im[crop[0]:crop[2], crop[1]:crop[3], ...]
        elif (im is not None) & (n_dims == 3):
            im = im[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], ...]
        if (lab is not None) & (n_dims == 2):
            lab = lab[crop[0]:crop[2], crop[1]:crop[3], ...]
        elif (lab is not None) & (n_dims == 3):
            lab = lab[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], ...]

        # loop over patches
        n_im = 0
        n_crop = (new_size / patch_shape).astype('int')
        for i in range(n_crop[0]):
            i *= patch_shape[0]
            for j in range(n_crop[1]):
                j *= patch_shape[1]

                if n_dims == 2:

                    # crop volumes
                    if lab is not None:
                        temp_la = lab[i:i+patch_shape[0], j:j+patch_shape[1], ...]
                    else:
                        temp_la = None
                    if im is not None:
                        temp_im = im[i:i + patch_shape[0], j:j + patch_shape[1], ...]
                    else:
                        temp_im = None

                    # write patches
                    if temp_la is not None:
                        if full_background | (not (temp_la == 0).all()):
                            n_im += 1
                            utils.save_volume(temp_la, aff_lab, h_lab, os.path.join(labels_result_dir,
                                              os.path.basename(path_label.replace('.nii.gz', '_%d.nii.gz' % n_im))))
                            if temp_im is not None:
                                utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                                  os.path.basename(path_image.replace('.nii.gz', '_%d.nii.gz' % n_im))))
                    else:
                        utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                          os.path.basename(path_image.replace('.nii.gz', '_%d.nii.gz' % n_im))))

                elif n_dims == 3:
                    for k in range(n_crop[2]):
                        k *= patch_shape[2]

                        # crop volumes
                        if lab is not None:
                            temp_la = lab[i:i + patch_shape[0], j:j + patch_shape[1], k:k+patch_shape[2], ...]
                        else:
                            temp_la = None
                        if im is not None:
                            temp_im = im[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2], ...]
                        else:
                            temp_im = None

                        # write patches
                        if temp_la is not None:
                            if full_background | (not (temp_la == 0).all()):
                                n_im += 1
                                utils.save_volume(temp_la, aff_lab, h_lab, os.path.join(labels_result_dir,
                                                  os.path.basename(path_label.replace('.nii.gz', '_%d.nii.gz' % n_im))))
                                if temp_im is not None:
                                    utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                                      os.path.basename(path_image.replace('.nii.gz',
                                                                                          '_%d.nii.gz' % n_im))))
                        else:
                            utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                              os.path.basename(path_image.replace('.nii.gz', '_%d.nii.gz' % n_im))))

        if remove_after_dividing:
            if path_image is not None:
                os.remove(path_image)
            if path_label is not None:
                os.remove(path_label)
