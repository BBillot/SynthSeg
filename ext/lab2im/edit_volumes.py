"""
This file contains functions to edit/preprocess volumes (i.e. not tensors!).
These functions are sorted in five categories:
1- volume editing: this can be applied to any volume (i.e. images or label maps). It contains:
        -mask_volume
        -rescale_volume
        -crop_volume
        -crop_volume_around_region
        -crop_volume_with_idx
        -pad_volume
        -flip_volume
        -resample_volume
        -resample_volume_like
        -get_ras_axes
        -align_volume_to_ref
        -blur_volume
2- label map editing: can be applied to label maps only. It contains:
        -correct_label_map
        -mask_label_map
        -smooth_label_map
        -erode_label_map
        -get_largest_connected_component
        -compute_hard_volumes
        -compute_distance_map
3- editing all volumes in a folder: functions are more or less the same as 1, but they now apply to all the volumes
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
        -niftyreg_images_in_dir
        -upsample_anisotropic_images
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
5- dataset editing: functions for editing datasets (i.e. images with corresponding label maps). It contains:
        -check_images_and_labels
        -crop_dataset_to_minimum_size
        -subdivide_dataset_to_patches


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
import os
import csv
import shutil
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from scipy.ndimage.filters import convolve
from scipy.ndimage import label as scipy_label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

# project imports
from ext.lab2im import utils
from ext.lab2im.layers import GaussianBlur, ConvertLabels
from ext.lab2im.edit_tensors import blurring_sigma_for_downsampling


# ---------------------------------------------------- edit volume -----------------------------------------------------

def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False, masking_value=0,
                return_mask=False, return_copy=True):
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
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = utils.get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = new_volume >= threshold
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
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume


def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2, max_percentile=98, use_positive_only=False):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None, return_crop_idx=False, mode='center'):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. The cropping margin is applied on both sides.
    Can be an int, sequence or 1d numpy array of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None, and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = utils.get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = utils.reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = utils.reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum([min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                                      np.array(vol_shape)[:n_dims])
        elif mode == 'random':
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)


def crop_volume_around_region(volume,
                              mask=None,
                              masking_labels=None,
                              threshold=0.1,
                              margin=0,
                              cropping_shape=None,
                              cropping_shape_div_by=None,
                              aff=None,
                              overflow='strict'):
    """Crop a volume around a specific region.
    This region is defined by a mask obtained by either:
    1) directly specifying it as input (see mask)
    2) keeping a set of label values if the volume is a label map (see masking_labels).
    3) thresholding the input volume (see threshold)
    The cropping region is defined by the bounding box of the mask, which we can further modify by either:
    1) extending it by a margin (see margin)
    2) providing a specific cropping shape, in this case the cropping region will be centered around the bounding box
    (see cropping_shape).
    3) extending it to a shape that is divisible by a given number. Again, the cropping region will be centered around
    the bounding box (see cropping_shape_div_by).
    Finally, if the size of the cropping region has been modified, and that this modified size overflows out of the
    image (e.g. because the center of the mask is close to the edge), we can either:
    1) stick to the valid image space (the size of the modified cropping region won't be respected)
    2) shift the cropping region so that it lies on the valid image space, and if it still overflows, then we restrict
    to the valid image space.
    3) pad the image with zeros, such that the cropping region is not ill-defined anymore.
    3) shift the cropping region to the valida image space, and if it still overflows, then we pad with zeros.
    :param volume: a 2d or 3d numpy array
    :param mask: (optional) mask of region to crop around. Must be same size as volume. Can either be boolean or 0/1.
    If no mask is given, it will be computed by either thresholding the input volume or using masking_labels.
    :param masking_labels: (optional) if mask is None, and if the volume is a label map, it can be cropped around a
    set of labels specified in masking_labels, which can either be a single int, a sequence or a 1d numpy array.
    :param threshold: (optional) if mask amd masking_labels are None, lower bound to determine values to crop around.
    :param margin: (optional) add margin around mask
    :param cropping_shape: (optional) shape to which the input volumes must be cropped. Volumes are padded around the
    centre of the above-defined mask is they are too small for the given shape. Can be an integer or sequence.
    Cannot be given at the same time as margin or cropping_shape_div_by.
    :param cropping_shape_div_by: (optional) makes sure the shape of the cropped region is divisible by the provided
    number. If it is not, then we enlarge the cropping area. If the enlarged area is too big for the input volume, we
    pad it with 0. Must be an integer. Cannot be given at the same time as margin or cropping_shape.
    :param aff: (optional) if specified, this function returns an updated affine matrix of the volume after cropping.
    :param overflow: (optional) how to proceed when the cropping region overflows outside the initial image space.
    Can either be 'strict' (default), 'shift-strict', 'padding', 'shift-padding.
    :return: the cropped volume, the cropping indices (in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...]),
    and the updated affine matrix if aff is not None.
    """

    assert not ((margin > 0) & (cropping_shape is not None)), "margin and cropping_shape can't be given together."
    assert not ((margin > 0) & (cropping_shape_div_by is not None)), \
        "margin and cropping_shape_div_by can't be given together."
    assert not ((cropping_shape_div_by is not None) & (cropping_shape is not None)), \
        "cropping_shape_div_by and cropping_shape can't be given together."

    new_vol = volume.copy()
    n_dims, n_channels = utils.get_dims(new_vol.shape)
    vol_shape = np.array(new_vol.shape[:n_dims])

    # mask ROIs for cropping
    if mask is None:
        if masking_labels is not None:
            _, mask = mask_label_map(new_vol, masking_values=masking_labels, return_mask=True)
        else:
            mask = new_vol > threshold

    # find cropping indices
    if np.any(mask):

        indices = np.nonzero(mask)
        min_idx = np.array([np.min(idx) for idx in indices])
        max_idx = np.array([np.max(idx) for idx in indices])
        intermediate_vol_shape = max_idx - min_idx

        if (margin == 0) & (cropping_shape is None) & (cropping_shape_div_by is None):
            cropping_shape = intermediate_vol_shape
        if margin:
            cropping_shape = intermediate_vol_shape + 2 * margin
        elif cropping_shape is not None:
            cropping_shape = np.array(utils.reformat_to_list(cropping_shape, length=n_dims))
        elif cropping_shape_div_by is not None:
            cropping_shape = [utils.find_closest_number_divisible_by_m(s, cropping_shape_div_by, answer_type='higher')
                              for s in intermediate_vol_shape]

        min_idx = min_idx - np.int32(np.ceil((cropping_shape - intermediate_vol_shape) / 2))
        max_idx = max_idx + np.int32(np.floor((cropping_shape - intermediate_vol_shape) / 2))
        min_overflow = np.abs(np.minimum(min_idx, 0))
        max_overflow = np.maximum(max_idx - vol_shape, 0)

        if 'strict' in overflow:
            min_overflow = np.zeros_like(min_overflow)
            max_overflow = np.zeros_like(min_overflow)

        if overflow == 'shift-strict':
            min_idx -= max_overflow
            max_idx += min_overflow

        if overflow == 'shift-padding':
            for ii in range(n_dims):
                # no need to do anything if both min/max_overflow are 0 (no padding/shifting required at all)
                # or if both are positive, because in this case we don't shift at all and we pad directly
                if (min_overflow[ii] > 0) & (max_overflow[ii] == 0):
                    max_idx_new = max_idx[ii] + min_overflow[ii]
                    if max_idx_new <= vol_shape[ii]:
                        max_idx[ii] = max_idx_new
                        min_overflow[ii] = 0
                    else:
                        min_overflow[ii] = min_overflow[ii] - (vol_shape[ii] - max_idx[ii])
                        max_idx[ii] = vol_shape[ii]
                elif (min_overflow[ii] == 0) & (max_overflow[ii] > 0):
                    min_idx_new = min_idx[ii] - max_overflow[ii]
                    if min_idx_new >= 0:
                        min_idx[ii] = min_idx_new
                        max_overflow[ii] = 0
                    else:
                        max_overflow[ii] = max_overflow[ii] - min_idx[ii]
                        min_idx[ii] = 0

        # crop volume if necessary
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, vol_shape)
        cropping = np.concatenate([min_idx, max_idx])
        if np.any(cropping[:3] > 0) or np.any(cropping[3:] != vol_shape):
            if n_dims == 3:
                new_vol = new_vol[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...]
            elif n_dims == 2:
                new_vol = new_vol[cropping[0]:cropping[2], cropping[1]:cropping[3], ...]
            else:
                raise ValueError('cannot crop volumes with more than 3 dimensions')

        # pad volume if necessary
        if np.any(min_overflow > 0) | np.any(max_overflow > 0):
            pad_margins = tuple([(min_overflow[i], max_overflow[i]) for i in range(n_dims)])
            pad_margins = tuple(list(pad_margins) + [(0, 0)]) if n_channels > 1 else pad_margins
            new_vol = np.pad(new_vol, pad_margins, mode='constant', constant_values=0)

    # if there's nothing to crop around, we return the input as is
    else:
        min_idx = min_overflow = np.zeros(3)
        cropping = None

    # return results
    if aff is not None:
        if n_dims == 2:
            min_idx = np.append(min_idx, 0)
            min_overflow = np.append(min_overflow, 0)
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ min_idx
        aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_overflow
        return new_vol, cropping, aff
    else:
        return new_vol, cropping


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: cropping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :param return_pad_idx: (optional) the pad_idx corresponds to the indices where we should crop the resulting
    padded image (ie the output of this function) to go back to the original volume (ie the input of this function).
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = utils.get_dims(vol_shape)
    padding_shape = utils.reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: flipped volume
    """

    new_volume = volume.copy() if return_copy else volume
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
    return np.flip(new_volume, axis=axis)


def resample_volume(volume, aff, new_vox_size, interpolation='linear', blur=True):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :param interpolation: (optional) type of interpolation. Can be 'linear' or 'nearest'. Default is 'linear'.
    :param blur: (optional) whether to blur before resampling to avoid aliasing effects.
    Only used if the input volume is downsampled. Default is True.
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas) if blur else volume

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator((x, y, z), volume_filt, method=interpolation)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def resample_volume_like(vol_ref, aff_ref, vol_flo, aff_flo, interpolation='linear'):
    """This function reslices a floating image to the space of a reference image
    :param vol_ref: a numpy array with the reference volume
    :param aff_ref: affine matrix of the reference volume
    :param vol_flo: a numpy array with the floating volume
    :param aff_flo: affine matrix of the floating volume
    :param interpolation: (optional) type of interpolation. Can be 'linear' or 'nearest'. Default is 'linear'.
    :return: resliced volume
    """

    T = np.matmul(np.linalg.inv(aff_flo), aff_ref)

    xf = np.arange(0, vol_flo.shape[0])
    yf = np.arange(0, vol_flo.shape[1])
    zf = np.arange(0, vol_flo.shape[2])

    my_interpolating_function = RegularGridInterpolator((xf, yf, zf), vol_flo, bounds_error=False, fill_value=0.0,
                                                        method=interpolation)

    xr = np.arange(0, vol_ref.shape[0])
    yr = np.arange(0, vol_ref.shape[1])
    zr = np.arange(0, vol_ref.shape[2])

    xrg, yrg, zrg = np.meshgrid(xr, yr, zr, indexing='ij', sparse=False)
    n = xrg.size
    xrg = xrg.reshape([n])
    yrg = yrg.reshape([n])
    zrg = zrg.reshape([n])
    bottom = np.ones_like(xrg)
    coords = np.stack([xrg, yrg, zrg, bottom])
    coords_new = np.matmul(T, coords)[:-1, :]
    result = my_interpolating_function((coords_new[0, :], coords_new[1, :], coords_new[2, :]))

    return result.reshape(vol_ref.shape)


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of length n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :param return_copy: (optional) whether to return the original volume or a copy. Default is copy.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = utils.get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


def blur_volume(volume, sigma, mask=None):
    """Blur volume with gaussian masks of given sigma.
    :param volume: 2d or 3d numpy array
    :param sigma: standard deviation of the gaussian kernels. Can be a number, a sequence or a 1d numpy array
    :param mask: (optional) numpy array of the same shape as volume to correct for edge blurring effects.
    Mask can be a boolean or numerical array. In the latter, the mask is computed by keeping all values above zero.
    :return: blurred volume
    """

    # initialisation
    new_volume = volume.copy()
    n_dims, _ = utils.get_dims(new_volume.shape)
    sigma = utils.reformat_to_list(sigma, length=n_dims, dtype='float')

    # blur image
    new_volume = gaussian_filter(new_volume, sigma=sigma, mode='nearest')  # nearest refers to edge padding

    # correct edge effect if mask is not None
    if mask is not None:
        assert new_volume.shape == mask.shape, 'volume and mask should have the same dimensions: ' \
                                               'got {0} and {1}'.format(new_volume.shape, mask.shape)
        mask = (mask > 0) * 1.0
        blurred_mask = gaussian_filter(mask, sigma=sigma, mode='nearest')
        new_volume = new_volume / (blurred_mask + 1e-6)
        new_volume[mask == 0] = 0

    return new_volume


# --------------------------------------------------- edit label map ---------------------------------------------------

def correct_label_map(labels, list_incorrect_labels, list_correct_labels=None, use_nearest_label=False,
                      remove_zero=False, smooth=False):
    """This function corrects specified label values in a label map by either a list of given values, or by the nearest
    label.
    :param labels: a 2d or 3d label map
    :param list_incorrect_labels: list of all label values to correct (eg [1, 2, 3]). Can also be a path to such a list.
    :param list_correct_labels: (optional) list of correct label values to replace the incorrect ones.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list within
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param use_nearest_label: (optional) whether to correct the incorrect label values with the nearest labels.
    :param remove_zero: (optional) if use_nearest_label is True, set to True not to consider zero among the potential
    candidates for the nearest neighbour. -1 will be returned when no solution are possible.
    :param smooth: (optional) whether to smooth the corrected label map
    :return: corrected label map
    """

    assert (list_correct_labels is not None) | use_nearest_label, \
        'please provide a list of correct labels, or set use_nearest_label to True.'
    assert (list_correct_labels is None) | (not use_nearest_label), \
        'cannot provide a list of correct values and set use_nearest_label to True'

    # initialisation
    new_labels = labels.copy()
    list_incorrect_labels = utils.reformat_to_list(utils.load_array_if_path(list_incorrect_labels))
    volume_labels = np.unique(labels)
    n_dims, _ = utils.get_dims(labels.shape)

    # use list of correct values
    if list_correct_labels is not None:
        list_correct_labels = utils.reformat_to_list(utils.load_array_if_path(list_correct_labels))

        # loop over label values
        for incorrect_label, correct_label in zip(list_incorrect_labels, list_correct_labels):
            if incorrect_label in volume_labels:

                # only one possible value to replace with
                if isinstance(correct_label, (int, float, np.int64, np.int32, np.int16, np.int8)):
                    incorrect_voxels = np.where(labels == incorrect_label)
                    new_labels[incorrect_voxels] = correct_label

                # several possibilities
                elif isinstance(correct_label, (tuple, list)):

                    # make sure at least one correct label is present
                    if not any([lab in volume_labels for lab in correct_label]):
                        print('no correct values found in volume, please adjust: '
                              'incorrect: {}, correct: {}'.format(incorrect_label, correct_label))

                    # crop around incorrect label until we find incorrect labels
                    correct_label_not_found = True
                    margin_mult = 1
                    tmp_labels = None
                    crop = None
                    while correct_label_not_found:
                        tmp_labels, crop = crop_volume_around_region(labels,
                                                                     masking_labels=incorrect_label,
                                                                     margin=10 * margin_mult)
                        correct_label_not_found = not any([lab in np.unique(tmp_labels) for lab in correct_label])
                        margin_mult += 1

                    # calculate distance maps for all new label candidates
                    incorrect_voxels = np.where(tmp_labels == incorrect_label)
                    distance_map_list = [distance_transform_edt(tmp_labels != lab) for lab in correct_label]
                    distances_correct = np.stack([dist[incorrect_voxels] for dist in distance_map_list])

                    # select nearest values and use them to correct label map
                    idx_correct_lab = np.argmin(distances_correct, axis=0)
                    incorrect_voxels = tuple([incorrect_voxels[i] + crop[i] for i in range(n_dims)])
                    new_labels[incorrect_voxels] = np.array(correct_label)[idx_correct_lab]

    # use nearest label
    else:

        # loop over label values
        for incorrect_label in list_incorrect_labels:
            if incorrect_label in volume_labels:

                # loop around regions
                components, n_components = scipy_label(labels == incorrect_label)
                loop_info = utils.LoopInfo(n_components + 1, 100, 'correcting')
                for i in range(1, n_components + 1):
                    loop_info.update(i)

                    # crop each region
                    _, crop = crop_volume_around_region(components, masking_labels=i, margin=1)
                    tmp_labels = crop_volume_with_idx(labels, crop)
                    tmp_new_labels = crop_volume_with_idx(new_labels, crop)

                    # list all possible correct labels
                    correct_labels = np.unique(tmp_labels)
                    for il in list_incorrect_labels:
                        correct_labels = np.delete(correct_labels, np.where(correct_labels == il))
                    if remove_zero:
                        correct_labels = np.delete(correct_labels, np.where(correct_labels == 0))

                    # replace incorrect voxels by new value
                    incorrect_voxels = np.where(tmp_labels == incorrect_label)
                    if len(correct_labels) == 0:
                        tmp_new_labels[incorrect_voxels] = -1
                    else:
                        if len(correct_labels) == 1:
                            idx_correct_lab = np.zeros(len(incorrect_voxels[0]), dtype='int32')
                        else:
                            distance_map_list = [distance_transform_edt(tmp_labels != lab) for lab in correct_labels]
                            distances_correct = np.stack([dist[incorrect_voxels] for dist in distance_map_list])
                            idx_correct_lab = np.argmin(distances_correct, axis=0)
                        tmp_new_labels[incorrect_voxels] = np.array(correct_labels)[idx_correct_lab]

                    # paste back
                    if n_dims == 2:
                        new_labels[crop[0]:crop[2], crop[1]:crop[3], ...] = tmp_new_labels
                    else:
                        new_labels[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], ...] = tmp_new_labels

    # smoothing
    if smooth:
        kernel = np.ones(tuple([3] * n_dims))
        new_labels = smooth_label_map(new_labels, kernel)

    return new_labels


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
    for value in utils.reformat_to_list(masking_values):
        mask = mask | (labels == value)
    masked_labels[np.logical_not(mask)] = masking_value

    if return_mask:
        mask = mask * 1
        return masked_labels, mask
    else:
        return masked_labels


def smooth_label_map(labels, kernel, labels_list=None, print_progress=0):
    """This function smooth an input label map by replacing each voxel by the value of its most numerous neighbour.
    :param labels: input label map
    :param kernel: kernel when counting neighbours. Must contain only zeros or ones.
    :param labels_list: list of label values to smooth. Defaults is None, where all labels are smoothed.
    :param print_progress: (optional) If not 0, interval at which to print the number of processed labels.
    :return: smoothed label map
    """
    # get info
    labels_shape = labels.shape
    unique_labels = np.unique(labels).astype('int32')
    if labels_list is None:
        labels_list = unique_labels
        new_labels = mask_new_labels = None
    else:
        labels_to_keep = [lab for lab in unique_labels if lab not in labels_list]
        new_labels, mask_new_labels = mask_label_map(labels, labels_to_keep, return_mask=True)

    # loop through label values
    count = np.zeros(labels_shape)
    labels_smoothed = np.zeros(labels_shape, dtype='int')
    loop_info = utils.LoopInfo(len(labels_list), print_progress, 'smoothing')
    for la, label in enumerate(labels_list):
        if print_progress:
            loop_info.update(la)

        # count neighbours with same value
        mask = (labels == label) * 1
        n_neighbours = convolve(mask, kernel)

        # update label map and maximum neighbour counts
        idx = n_neighbours > count
        count[idx] = n_neighbours[idx]
        labels_smoothed[idx] = label
        labels_smoothed = labels_smoothed.astype('int32')

    if new_labels is None:
        new_labels = labels_smoothed
    else:
        new_labels = np.where(mask_new_labels, new_labels, labels_smoothed)

    return new_labels


def erode_label_map(labels, labels_to_erode, erosion_factors=1., gpu=False, model=None, return_model=False):
    """Erode a given set of label values within a label map.
    :param labels: a 2d or 3d label map
    :param labels_to_erode: list of label values to erode
    :param erosion_factors: (optional) list of erosion factors to use for each label. If values are integers, normal
    erosion applies. If float, we first 1) blur a mask of the corresponding label value, and 2) use the erosion factor
    as a threshold in the blurred mask.
    If erosion_factors is a single value, the same factor will be applied to all labels.
    :param gpu: (optional) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param model: (optional) gpu model for blurring masks (if erosion factors are floats)
    :param return_model: (optional) whether to return the gpu blurring model
    :return: eroded label map, and gpu blurring model is return_model is True.
    """
    # reformat labels_to_erode and erode
    new_labels = labels.copy()
    labels_to_erode = utils.reformat_to_list(labels_to_erode)
    erosion_factors = utils.reformat_to_list(erosion_factors, length=len(labels_to_erode))
    labels_shape = list(new_labels.shape)
    n_dims, _ = utils.get_dims(labels_shape)

    # loop over labels to erode
    for label_to_erode, erosion_factor in zip(labels_to_erode, erosion_factors):

        assert erosion_factor > 0, 'all erosion factors should be strictly positive, had {}'.format(erosion_factor)

        # get mask of current label value
        mask = (new_labels == label_to_erode)

        # erode as usual if erosion factor is int
        if int(erosion_factor) == erosion_factor:
            erode_struct = utils.build_binary_structure(int(erosion_factor), n_dims)
            eroded_mask = binary_erosion(mask, erode_struct)

        # blur mask and use erosion factor as a threshold if float
        else:
            if gpu:
                if model is None:
                    mask_in = KL.Input(shape=labels_shape + [1], dtype='float32')
                    blurred_mask = GaussianBlur([1] * 3)(mask_in)
                    model = Model(inputs=mask_in, outputs=blurred_mask)
                eroded_mask = model.predict(utils.add_axis(np.float32(mask), axis=[0, -1]))
            else:
                eroded_mask = blur_volume(np.array(mask, dtype='float32'), 1)
            eroded_mask = np.squeeze(eroded_mask) > erosion_factor

        # crop label map and mask around values to change
        mask = mask & np.logical_not(eroded_mask)
        cropped_lab_mask, cropping = crop_volume_around_region(mask, margin=3)
        cropped_labels = crop_volume_with_idx(new_labels, cropping)

        # calculate distance maps for all labels in cropped_labels
        labels_list = np.unique(cropped_labels)
        labels_list = labels_list[labels_list != label_to_erode]
        list_dist_maps = [distance_transform_edt(np.logical_not(cropped_labels == la)) for la in labels_list]
        candidate_distances = np.stack([dist[cropped_lab_mask] for dist in list_dist_maps])

        # select nearest value and put cropped labels back to full label map
        idx_correct_lab = np.argmin(candidate_distances, axis=0)
        cropped_labels[cropped_lab_mask] = np.array(labels_list)[idx_correct_lab]
        if n_dims == 2:
            new_labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...] = cropped_labels
        elif n_dims == 3:
            new_labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...] = cropped_labels

        if return_model:
            return new_labels, model
        else:
            return new_labels


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()


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
    for these labels only. Default is None, where all positive values are considered.
    :param crop_margin: (optional) margin with which to crop the input label maps around the labels for which we
    want to compute the distance maps.
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
    loop_info = utils.LoopInfo(len(path_images), 10, 'masking', True)
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        loop_info.update(idx)

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
                          min_percentile=2, max_percentile=98, use_positive_only=True,
                          recompute=True):
    """This function linearly rescales all volumes in image_dir between new_min and new_max.
    :param image_dir: path of directory with images to rescale
    :param result_dir: path of directory where rescaled images will be writen
    :param new_min: (optional) minimum value for the rescaled images.
    :param new_max: (optional) maximum value for the rescaled images.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    loop_info = utils.LoopInfo(len(path_images), 10, 'rescaling', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

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
    loop_info = utils.LoopInfo(len(path_images), 10, 'cropping', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

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
    loop_info = utils.LoopInfo(len(path_images), 10, 'cropping', True)
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        loop_info.update(idx)

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
    :param recompute: (optional) whether to recompute result files even if they already exist
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
    loop_info = utils.LoopInfo(len(path_images), 10, 'padding', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

        # pad map
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im, aff = pad_volume(im, max_shape, padding_value, aff)
            utils.save_volume(im, aff, h, path_result)

    return max_shape


def flip_images_in_dir(image_dir, result_dir, axis=None, direction=None, recompute=True):
    """Flip all images in a directory along a specified axis.
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
    loop_info = utils.LoopInfo(len(path_images), 10, 'flipping', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

        # flip image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im = flip_volume(im, axis=axis, direction=direction, aff=aff)
            utils.save_volume(im, aff, h, path_result)


def align_images_in_dir(image_dir, result_dir, aff_ref=None, path_ref=None, recompute=True):
    """This function aligns all images in image_dir to a reference orientation (axes and directions).
    This reference orientation can be directly provided as an affine matrix, or can be specified by a reference volume.
    If neither are provided, the reference orientation is assumed to be an identity matrix.
    :param image_dir: path of directory with images to align
    :param result_dir: path of directory where flipped images will be writen
    :param aff_ref: (optional) reference affine matrix. Can be a numpy array, or the path to such array.
    :param path_ref: (optional) path of a volume to which all images will be aligned. Can also be the path to a folder
    with as many images as in image_dir, in which case each image in image_dir is aligned to its counterpart in path_ref
    (they are matched by sorting order).
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)
    path_images = utils.list_images_in_folder(image_dir)

    # read reference affine matrix
    if path_ref is not None:
        assert aff_ref is None, 'cannot provide aff_ref and path_ref together.'
        basename = os.path.basename(path_ref)
        if ('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename):
            _, aff_ref, _ = utils.load_volume(path_ref, im_only=False)
            path_refs = [None] * len(path_images)
        else:
            path_refs = utils.list_images_in_folder(path_ref)
    elif aff_ref is not None:
        aff_ref = utils.load_array_if_path(aff_ref)
        path_refs = [None] * len(path_images)
    else:
        aff_ref = np.eye(4)
        path_refs = [None] * len(path_images)

    # loop over images
    loop_info = utils.LoopInfo(len(path_images), 10, 'aligning', True)
    for idx, (path_image, path_ref) in enumerate(zip(path_images, path_refs)):
        loop_info.update(idx)

        # align image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            if path_ref is not None:
                _, aff_ref, _ = utils.load_volume(path_ref, im_only=False)
            im, aff = align_volume_to_ref(im, aff, aff_ref=aff_ref, return_aff=True)
            utils.save_volume(im, aff, h, path_result)


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
    loop_info = utils.LoopInfo(len(path_images), 10, 'correcting', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

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
    Can be a number (isotropic blurring), or a sequence with the same length as the number of dimensions of images.
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
    loop_info = utils.LoopInfo(len(path_images), 10, 'blurring', True)
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        loop_info.update(idx)

        # load image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, im_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_image, return_volume=True)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
                assert mask.shape == im.shape, 'mask and image should have the same shape'
            else:
                mask = None

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    inputs = [KL.Input(shape=im_shape + [1])]
                    sigma = utils.reformat_to_list(sigma, length=n_dims)
                    if mask is None:
                        image = GaussianBlur(sigma=sigma)(inputs[0])
                    else:
                        inputs.append(KL.Input(shape=im_shape + [1], dtype='float32'))
                        image = GaussianBlur(sigma=sigma, use_mask=True)(inputs)
                    model = Model(inputs=inputs, outputs=image)
                if mask is None:
                    im = np.squeeze(model.predict(utils.add_axis(im, axis=[0, -1])))
                else:
                    im = np.squeeze(model.predict([utils.add_axis(im, [0, -1]), utils.add_axis(mask, [0, -1])]))
            else:
                im = blur_volume(im, sigma, mask=mask)
            utils.save_volume(im, aff, h, path_result)


def create_mutlimodal_images(list_channel_dir, result_dir, recompute=True):
    """This function forms multimodal images by stacking channels located in different folders.
    :param list_channel_dir: list of all directories, each containing the same channel for all images.
    Channels are matched between folders by sorting order.
    :param result_dir: path of directory where multimodal images will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    assert isinstance(list_channel_dir, (list, tuple)), 'list_channel_dir should be a list or a tuple'

    # gather path of all images for all channels
    list_channel_paths = [utils.list_images_in_folder(d) for d in list_channel_dir]
    n_images = len(list_channel_paths[0])
    n_channels = len(list_channel_dir)
    for channel_paths in list_channel_paths:
        if len(channel_paths) != n_images:
            raise ValueError('all directories should have the same number of files')

    # loop over images
    loop_info = utils.LoopInfo(n_images, 10, 'processing', True)
    for idx in range(n_images):
        loop_info.update(idx)

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


def convert_images_in_dir_to_nifty(image_dir, result_dir, aff=None, ref_aff_dir=None, recompute=True):
    """Converts all images in image_dir to nifty format.
    :param image_dir: path of directory with images to convert
    :param result_dir: path of directory where converted images will be writen
    :param aff: (optional) affine matrix in homogeneous coordinates with which to write the images.
    Can also be 'FS' to write images with FreeSurfer typical affine matrix.
    :param ref_aff_dir: (optional) alternatively to providing a fixed aff, different affine matrices can be used for
    each image in image_dir by matching them to corresponding volumes contained in ref_aff_dir.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # list images
    path_images = utils.list_images_in_folder(image_dir)
    if ref_aff_dir is not None:
        path_ref_images = utils.list_images_in_folder(ref_aff_dir)
    else:
        path_ref_images = [None] * len(path_images)

    # loop over images
    loop_info = utils.LoopInfo(len(path_images), 10, 'converting', True)
    for idx, (path_image, path_ref) in enumerate(zip(path_images, path_ref_images)):
        loop_info.update(idx)

        # convert images to nifty format
        path_result = os.path.join(result_dir, os.path.basename(utils.strip_extension(path_image))) + '.nii.gz'
        if (not os.path.isfile(path_result)) | recompute:
            if utils.get_image_extension(path_image) == 'nii.gz':
                shutil.copy2(path_image, path_result)
            else:
                im, tmp_aff, h = utils.load_volume(path_image, im_only=False)
                if aff is not None:
                    tmp_aff = aff
                elif path_ref is not None:
                    _, tmp_aff, h = utils.load_volume(path_ref, im_only=False)
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
    loop_info = utils.LoopInfo(len(path_images), 10, 'converting', True)
    for idx, (path_image, path_reference) in enumerate(zip(path_images, path_references)):
        loop_info.update(idx)

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
    loop_info = utils.LoopInfo(len(path_images), 10, 'processing', True)
    for idx, path_image in enumerate(path_images):
        loop_info.update(idx)

        # build path_result
        path_im_result_dir = os.path.join(result_dir, utils.strip_extension(os.path.basename(path_image)))
        path_samseg_result = os.path.join(path_im_result_dir, 'seg.mgz')
        if keep_segm_only:
            path_result = os.path.join(result_dir, utils.strip_extension(os.path.basename(path_image)) + '_seg.mgz')
        else:
            path_result = path_samseg_result

        # run samseg
        if (not os.path.isfile(path_result)) | recompute:
            cmd = utils.mkcmd(path_samseg, '-i', path_image, '-o', path_im_result_dir, '--threads', threads)
            if atlas_dir is not None:
                cmd = utils.mkcmd(cmd, '-a', atlas_dir)
            os.system(cmd)

        # move segmentation to result_dir if necessary
        if keep_segm_only:
            if os.path.isfile(path_samseg_result):
                shutil.move(path_samseg_result, path_result)
            if os.path.isdir(path_im_result_dir):
                shutil.rmtree(path_im_result_dir)


def niftyreg_images_in_dir(image_dir,
                           reference_dir,
                           nifty_reg_function='reg_resample',
                           input_transformation_dir=None,
                           result_dir=None,
                           result_transformation_dir=None,
                           interpolation=None,
                           same_floating=False,
                           same_reference=False,
                           same_transformation=False,
                           path_nifty_reg='/home/benjamin/Softwares/niftyreg-gpu/build/reg-apps',
                           recompute=True):
    """This function launches one of niftyreg functions (reg_aladin, reg_f3d, reg_resample) on all images contained
    in image_dir.
    :param image_dir: path of directory with images to register. Can also be a single image, in that case set
    same_floating to True.
    :param reference_dir: path of directory with reference images. If same_reference is false, references and images are
    matched by sorting order. This can also be the path to a single image that will be used as reference for all images
    im image_dir (set same_reference to True in that case).
    :param nifty_reg_function: (optional) name of the niftyreg function to use. Can be 'reg_aladin', 'reg_f3d', or
    'reg_resample'. Default is 'reg_resample'.
    :param input_transformation_dir: (optional) path of a directory containing all the input transformation (for
    reg_resample, or reg_f3d). Can also be the path to a single transformation that will be used for all images
    in image_dir (set same_transformation to True in that case).
    :param result_dir: path of directory where output images will be writen.
    :param result_transformation_dir: path of directory where resulting transformations will be writen (for
    reg_aladin and reg_f3d).
    :param interpolation: (optional) integer describing the order of the interpolation to apply (0 = nearest neighbours)
    :param same_floating: (optional) set to true if only one image is used as floating image.
    :param same_reference: (optional) whether to use a single image as reference for all input images.
    :param same_transformation: (optional) whether to apply the same transformation to all floating images.
    :param path_nifty_reg: (optional) path of the folder containing nifty-reg functions
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dirs
    if result_dir is not None:
        utils.mkdir(result_dir)
    if result_transformation_dir is not None:
        utils.mkdir(result_transformation_dir)

    nifty_reg = os.path.join(path_nifty_reg, nifty_reg_function)

    # list reference and floating images
    path_images = utils.list_images_in_folder(image_dir)
    path_references = utils.list_images_in_folder(reference_dir)
    if same_reference:
        path_references = utils.reformat_to_list(path_references, length=len(path_images))
    if same_floating:
        path_images = utils.reformat_to_list(path_images, length=len(path_references))
    assert len(path_references) == len(path_images), 'different number of files in image_dir and reference_dir'

    # list input transformations
    if input_transformation_dir is not None:
        if same_transformation:
            path_input_transfs = utils.reformat_to_list(input_transformation_dir, length=len(path_images))
        else:
            path_input_transfs = utils.list_files(input_transformation_dir)
            assert len(path_input_transfs) == len(path_images), 'different number of transformations and images'
    else:
        path_input_transfs = [None] * len(path_images)

    # define flag input trans
    if input_transformation_dir is not None:
        if nifty_reg_function == 'reg_aladin':
            flag_input_trans = '-inaff'
        elif nifty_reg_function == 'reg_f3d':
            flag_input_trans = '-aff'
        elif nifty_reg_function == 'reg_resample':
            flag_input_trans = '-trans'
        else:
            raise Exception('nifty_reg_function can only be "reg_aladin", "reg_f3d", or "reg_resample"')
    else:
        flag_input_trans = None

    # define flag result transformation
    if result_transformation_dir is not None:
        if nifty_reg_function == 'reg_aladin':
            flag_result_trans = '-aff'
        elif nifty_reg_function == 'reg_f3d':
            flag_result_trans = '-cpp'
        else:
            raise Exception('result_transformation_dir can only be used with "reg_aladin" or "reg_f3d"')
    else:
        flag_result_trans = None

    # loop over images
    loop_info = utils.LoopInfo(len(path_images), 10, 'processing', True)
    for idx, (path_image, path_ref, path_input_trans) in enumerate(zip(path_images,
                                                                       path_references,
                                                                       path_input_transfs)):
        loop_info.update(idx)

        # define path registered image
        name = os.path.basename(path_ref) if same_floating else os.path.basename(path_image)
        if result_dir is not None:
            path_result = os.path.join(result_dir, name)
            result_already_computed = os.path.isfile(path_result)
        else:
            path_result = None
            result_already_computed = True

        # define path resulting transformation
        if result_transformation_dir is not None:
            if nifty_reg_function == 'reg_aladin':
                path_result_trans = os.path.join(result_transformation_dir, utils.strip_extension(name) + '.txt')
                result_trans_already_computed = os.path.isfile(path_result_trans)
            else:
                path_result_trans = os.path.join(result_transformation_dir, name)
                result_trans_already_computed = os.path.isfile(path_result_trans)
        else:
            path_result_trans = None
            result_trans_already_computed = True

        if (not result_already_computed) | (not result_trans_already_computed) | recompute:

            # build main command
            cmd = utils.mkcmd(nifty_reg, '-ref', path_ref, '-flo', path_image, '-pad 0')

            # add options
            if path_result is not None:
                cmd = utils.mkcmd(cmd, '-res', path_result)
            if flag_input_trans is not None:
                cmd = utils.mkcmd(cmd, flag_input_trans, path_input_trans)
            if flag_result_trans is not None:
                cmd = utils.mkcmd(cmd, flag_result_trans, path_result_trans)
            if interpolation is not None:
                cmd = utils.mkcmd(cmd, '-inter', interpolation)

            # execute
            os.system(cmd)


def upsample_anisotropic_images(image_dir,
                                resample_image_result_dir,
                                resample_like_dir,
                                path_freesurfer='/usr/local/freesurfer/',
                                recompute=True):
    """This function takes as input a set of LR images and resample them to HR with respect to reference images.
    :param image_dir: path of directory with input images (only uni-modal images supported)
    :param resample_image_result_dir: path of directory where resampled images will be writen
    :param resample_like_dir: path of directory with reference images.
    :param path_freesurfer: (optional) path freesurfer home, as this function uses mri_convert
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(resample_image_result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert')

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    path_ref_images = utils.list_images_in_folder(resample_like_dir)
    assert len(path_images) == len(path_ref_images), \
        'the folders containing the images and their references are not the same size'

    # loop over images
    loop_info = utils.LoopInfo(len(path_images), 10, 'upsampling', True)
    for idx, (path_image, path_ref) in enumerate(zip(path_images, path_ref_images)):
        loop_info.update(idx)

        # upsample image
        _, _, n_dims, _, _, image_res = utils.get_volume_info(path_image, return_volume=False)
        path_im_upsampled = os.path.join(resample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_upsampled)) | recompute:
            cmd = utils.mkcmd(mri_convert, path_image, path_im_upsampled, '-rl', path_ref, '-odt float')
            os.system(cmd)

        path_dist_map = os.path.join(resample_image_result_dir, 'dist_map_' + os.path.basename(path_image))
        if (not os.path.isfile(path_dist_map)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            dist_map = np.meshgrid(*[np.arange(s) for s in im.shape], indexing='ij')
            tmp_dir = utils.strip_extension(path_im_upsampled) + '_meshes'
            utils.mkdir(tmp_dir)
            path_meshes_up = list()
            for (i, maps) in enumerate(dist_map):
                path_mesh = os.path.join(tmp_dir, '%s_' % i + os.path.basename(path_image))
                path_mesh_up = os.path.join(tmp_dir, 'up_%s_' % i + os.path.basename(path_image))
                utils.save_volume(maps, aff, h, path_mesh)
                cmd = utils.mkcmd(mri_convert, path_mesh, path_mesh_up, '-rl', path_im_upsampled, '-odt float')
                os.system(cmd)
                path_meshes_up.append(path_mesh_up)
            mesh_up_0, aff, h = utils.load_volume(path_meshes_up[0], im_only=False)
            mesh_up = np.stack([mesh_up_0] + [utils.load_volume(p) for p in path_meshes_up[1:]], -1)
            shutil.rmtree(tmp_dir)

            floor = np.floor(mesh_up)
            ceil = np.ceil(mesh_up)
            f_dist = mesh_up - floor
            c_dist = ceil - mesh_up
            dist = np.minimum(f_dist, c_dist) * utils.add_axis(image_res, axis=[0] * n_dims)
            dist = np.sqrt(np.sum(dist ** 2, axis=-1))
            utils.save_volume(dist, aff, h, path_dist_map)


def simulate_upsampled_anisotropic_images(image_dir,
                                          downsample_image_result_dir,
                                          resample_image_result_dir,
                                          data_res,
                                          labels_dir=None,
                                          downsample_labels_result_dir=None,
                                          slice_thickness=None,
                                          build_dist_map=False,
                                          path_freesurfer='/usr/local/freesurfer/',
                                          gpu=True,
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
    :param build_dist_map: (optional) whether to return the resampled images with an additional channel indicating the
    distance of each voxel to the nearest acquired voxel. Default is False.
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
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert')

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = [None] * len(path_images) if labels_dir is None else utils.list_images_in_folder(labels_dir)

    # initialisation
    _, _, n_dims, _, _, image_res = utils.get_volume_info(path_images[0], return_volume=False, aff_ref=np.eye(4))
    data_res = np.squeeze(utils.reformat_to_n_channels_array(data_res, n_dims, n_channels=1))
    slice_thickness = utils.reformat_to_list(slice_thickness, length=n_dims)

    # loop over images
    previous_model_input_shape = None
    model = None
    loop_info = utils.LoopInfo(len(path_images), 10, 'processing', True)
    for idx, (path_image, path_labels) in enumerate(zip(path_images, path_labels)):
        loop_info.update(idx)

        # downsample image
        path_im_downsampled = os.path.join(downsample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_downsampled)) | recompute:
            im, _, aff, n_dims, _, h, image_res = utils.get_volume_info(path_image, return_volume=True)
            im, aff_aligned = align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=n_dims)
            im_shape = list(im.shape[:n_dims])
            sigma = blurring_sigma_for_downsampling(image_res, data_res, thickness=slice_thickness)
            sigma = [0 if data_res[i] == image_res[i] else sigma[i] for i in range(n_dims)]

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    image_in = KL.Input(shape=im_shape + [1])
                    image = GaussianBlur(sigma=sigma)(image_in)
                    model = Model(inputs=image_in, outputs=image)
                im = np.squeeze(model.predict(utils.add_axis(im, axis=[0, -1])))
            else:
                im = blur_volume(im, sigma, mask=None)
            utils.save_volume(im, aff_aligned, h, path_im_downsampled)

            # downsample blurred image
            voxsize = ' '.join([str(r) for r in data_res])
            cmd = utils.mkcmd(mri_convert, path_im_downsampled, path_im_downsampled, '--voxsize', voxsize,
                              '-odt float -rt nearest')
            os.system(cmd)

        # downsample labels if necessary
        if path_labels is not None:
            path_lab_downsampled = os.path.join(downsample_labels_result_dir, os.path.basename(path_labels))
            if (not os.path.isfile(path_lab_downsampled)) | recompute:
                cmd = utils.mkcmd(mri_convert, path_labels, path_lab_downsampled, '-rl', path_im_downsampled,
                                  '-odt float -rt nearest')
                os.system(cmd)

        # upsample image
        path_im_upsampled = os.path.join(resample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_upsampled)) | recompute:
            cmd = utils.mkcmd(mri_convert, path_im_downsampled, path_im_upsampled, '-rl', path_image, '-odt float')
            os.system(cmd)

        if build_dist_map:
            path_dist_map = os.path.join(resample_image_result_dir, 'dist_map_' + os.path.basename(path_image))
            if (not os.path.isfile(path_dist_map)) | recompute:
                im, aff, h = utils.load_volume(path_im_downsampled, im_only=False)
                dist_map = np.meshgrid(*[np.arange(s) for s in im.shape], indexing='ij')
                tmp_dir = utils.strip_extension(path_im_downsampled) + '_meshes'
                utils.mkdir(tmp_dir)
                path_meshes_up = list()
                for (i, d_map) in enumerate(dist_map):
                    path_mesh = os.path.join(tmp_dir, '%s_' % i + os.path.basename(path_image))
                    path_mesh_up = os.path.join(tmp_dir, 'up_%s_' % i + os.path.basename(path_image))
                    utils.save_volume(d_map, aff, h, path_mesh)
                    cmd = utils.mkcmd(mri_convert, path_mesh, path_mesh_up, '-rl', path_image, '-odt float')
                    os.system(cmd)
                    path_meshes_up.append(path_mesh_up)
                mesh_up_0, aff, h = utils.load_volume(path_meshes_up[0], im_only=False)
                mesh_up = np.stack([mesh_up_0] + [utils.load_volume(p) for p in path_meshes_up[1:]], -1)
                shutil.rmtree(tmp_dir)

                floor = np.floor(mesh_up)
                ceil = np.ceil(mesh_up)
                f_dist = mesh_up - floor
                c_dist = ceil - mesh_up
                dist = np.minimum(f_dist, c_dist) * utils.add_axis(data_res, axis=[0] * n_dims)
                dist = np.sqrt(np.sum(dist ** 2, axis=-1))
                utils.save_volume(dist, aff, h, path_dist_map)


def check_images_in_dir(image_dir, check_values=False, keep_unique=True, max_channels=10, verbose=True):
    """Check if all volumes within the same folder share the same characteristics: shape, affine matrix, resolution.
    Also have option to check if all volumes have the same intensity values (useful for label maps).
    :return four lists, each containing the different values detected for a specific parameter among those to check."""

    # define information to check
    list_shape = list()
    list_aff = list()
    list_res = list()
    list_axes = list()
    if check_values:
        list_unique_values = list()
    else:
        list_unique_values = None

    # loop through files
    path_images = utils.list_images_in_folder(image_dir)
    loop_info = utils.LoopInfo(len(path_images), 10, 'checking', verbose) if verbose else None
    for idx, path_image in enumerate(path_images):
        if loop_info is not None:
            loop_info.update(idx)

        # get info
        im, shape, aff, n_dims, _, h, res = utils.get_volume_info(path_image, True, np.eye(4), max_channels)
        axes = get_ras_axes(aff, n_dims=n_dims).tolist()
        aff[:, np.arange(n_dims)] = aff[:, axes]
        aff = (np.int32(np.round(np.array(aff[:3, :3]), 2) * 100) / 100).tolist()
        res = (np.int32(np.round(np.array(res), 2) * 100) / 100).tolist()

        # add values to list if not already there
        if (shape not in list_shape) | (not keep_unique):
            list_shape.append(shape)
        if (aff not in list_aff) | (not keep_unique):
            list_aff.append(aff)
        if (res not in list_res) | (not keep_unique):
            list_res.append(res)
        if (axes not in list_axes) | (not keep_unique):
            list_axes.append(axes)
        if list_unique_values is not None:
            uni = np.unique(im).tolist()
            if (uni not in list_unique_values) | (not keep_unique):
                list_unique_values.append(uni)

    return list_shape, list_aff, list_res, list_axes, list_unique_values


# ----------------------------------------------- edit label maps in dir -----------------------------------------------

def correct_labels_in_dir(labels_dir, results_dir, incorrect_labels, correct_labels=None,
                          use_nearest_label=False, remove_zero=False, smooth=False, recompute=True):
    """This function corrects label values for all label maps in a folder with either
    - a list a given values,
    - or with the nearest label value.
    :param labels_dir: path of directory with input label maps
    :param results_dir: path of directory where corrected label maps will be writen
    :param incorrect_labels: list of all label values to correct (e.g. [1, 2, 3, 4]).
    :param correct_labels: (optional) list of correct label values to replace the incorrect ones.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list within
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param use_nearest_label: (optional) whether to correct the incorrect label values with the nearest labels.
    :param remove_zero: (optional) if use_nearest_label is True, set to True not to consider zero among the potential
    candidates for the nearest neighbour.
    :param smooth: (optional) whether to smooth the corrected label maps
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(results_dir)

    # prepare data files
    path_labels = utils.list_images_in_folder(labels_dir)
    loop_info = utils.LoopInfo(len(path_labels), 10, 'correcting', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

        # correct labels
        path_result = os.path.join(results_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_label, im_only=False, dtype='int32')
            im = correct_label_map(im, incorrect_labels, correct_labels, use_nearest_label, remove_zero, smooth)
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
    values_to_keep = utils.reformat_to_list(values_to_keep, load_as_numpy=True)

    # loop over labels
    path_labels = utils.list_images_in_folder(labels_dir)
    loop_info = utils.LoopInfo(len(path_labels), 10, 'masking', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

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


def smooth_labels_in_dir(labels_dir, result_dir, gpu=False, labels_list=None, connectivity=1, recompute=True):
    """Smooth all label maps in a folder by replacing each voxel by the value of its most numerous neighbours.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where smoothed label maps will be writen
    :param gpu: (optional) whether to use a gpu implementation for faster processing
    :param labels_list: (optional) if gpu is True, path of numpy array with all label values.
    Automatically computed if not provided.
    :param connectivity: (optional) connectivity to use when smoothing the label maps
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    utils.mkdir(result_dir)

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)

    if labels_list is not None:
        labels_list, _ = utils.get_list_labels(label_list=labels_list, FS_sort=True)

    if gpu:
        # initialisation
        previous_model_input_shape = None
        smoothing_model = None

        # loop over label maps
        loop_info = utils.LoopInfo(len(path_labels), 10, 'smoothing', True)
        for idx, path_label in enumerate(path_labels):
            loop_info.update(idx)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path_label))
            if (not os.path.isfile(path_result)) | recompute:
                labels, label_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_label, return_volume=True)
                if label_shape != previous_model_input_shape:
                    previous_model_input_shape = label_shape
                    smoothing_model = smoothing_gpu_model(label_shape, labels_list, connectivity)
                unique_labels = np.unique(labels).astype('int32')
                if labels_list is None:
                    smoothed_labels = smoothing_model.predict(utils.add_axis(labels))
                else:
                    labels_to_keep = [lab for lab in unique_labels if lab not in labels_list]
                    new_labels, mask_new_labels = mask_label_map(labels, labels_to_keep, return_mask=True)
                    smoothed_labels = np.squeeze(smoothing_model.predict(utils.add_axis(labels)))
                    smoothed_labels = np.where(mask_new_labels, new_labels, smoothed_labels)
                    mask_new_zeros = (labels > 0) & (smoothed_labels == 0)
                    smoothed_labels[mask_new_zeros] = labels[mask_new_zeros]
                utils.save_volume(smoothed_labels, aff, h, path_result, dtype='int32')

    else:
        # build kernel
        _, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])
        kernel = utils.build_binary_structure(connectivity, n_dims, shape=n_dims)

        # loop over label maps
        loop_info = utils.LoopInfo(len(path_labels), 10, 'smoothing', True)
        for idx, path in enumerate(path_labels):
            loop_info.update(idx)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path))
            if (not os.path.isfile(path_result)) | recompute:
                volume, aff, h = utils.load_volume(path, im_only=False)
                new_volume = smooth_label_map(volume, kernel, labels_list)
                utils.save_volume(new_volume, aff, h, path_result, dtype='int32')


def smoothing_gpu_model(label_shape, label_list, connectivity=1):
    """This function builds a gpu model in keras with a tensorflow backend to smooth label maps.
    This model replaces each voxel of the input by the value of its most numerous neighbour.
    :param label_shape: shape of the label map
    :param label_list: list of all labels to consider
    :param connectivity: (optional) connectivity to use when smoothing the label maps
    :return: gpu smoothing model
    """

    # convert labels so values are in [0, ..., N-1] and use one hot encoding
    n_labels = label_list.shape[0]
    labels_in = KL.Input(shape=label_shape, name='lab_input', dtype='int32')
    labels = ConvertLabels(label_list)(labels_in)
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels)

    # count neighbouring voxels
    n_dims, _ = utils.get_dims(label_shape)
    k = utils.add_axis(utils.build_binary_structure(connectivity, n_dims, shape=n_dims), axis=[-1, -1])
    kernel = KL.Lambda(lambda x: tf.convert_to_tensor(k, dtype='float32'))([])
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_labels, axis=-1))(labels)
    labels = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[0], kernel])
    for i in range(1, n_labels):
        tmp = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[i], kernel])
        labels = KL.Lambda(lambda x: tf.concat([x[0], x[1]], -1))([labels, tmp])

    # take the argmax and convert labels to original values
    labels = KL.Lambda(lambda x: tf.math.argmax(x, -1))(labels)
    labels = ConvertLabels(np.arange(n_labels), label_list)(labels)
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
    :param gpu: (optional) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    utils.mkdir(result_dir)

    # loop over label maps
    model = None
    path_labels = utils.list_images_in_folder(labels_dir)
    loop_info = utils.LoopInfo(len(path_labels), 5, 'eroding', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

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
    """This function upsamples all label maps within a folder. Importantly, each label map is converted into probability
    maps for all label values, and all these maps are upsampled separately. The upsampled label maps are recovered by
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
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert')

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)
    labels_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_labels[0], max_channels=3)

    # build command
    target_res = utils.reformat_to_list(target_res, length=n_dims)
    post_cmd = '-voxsize ' + ' '.join([str(r) for r in target_res]) + ' -odt float'

    # load label list and corresponding LUT to make sure that labels go from 0 to N-1
    label_list, _ = utils.get_list_labels(path_label_list, labels_dir=path_labels, FS_sort=False)
    new_label_list = np.arange(len(label_list), dtype='int32')
    lut = utils.get_mapping_lut(label_list)

    # loop over label maps
    loop_info = utils.LoopInfo(len(path_labels), 5, 'upsampling', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)
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
                path_mask = os.path.join(indiv_label_dir, str(label) + '.nii.gz')
                path_mask_upsampled = os.path.join(upsample_indiv_label_dir, str(label) + '.nii.gz')
                if not os.path.isfile(path_mask):
                    mask = (labels == label) * 1.0
                    utils.save_volume(mask, aff, h, path_mask)
                if not os.path.isfile(path_mask_upsampled):
                    cmd = utils.mkcmd(mri_convert, path_mask, path_mask_upsampled, post_cmd)
                    os.system(cmd)

            # compute argmax of upsampled probability maps (upload them one at a time)
            probmax, aff, h = utils.load_volume(os.path.join(upsample_indiv_label_dir, '0.nii.gz'), im_only=False)
            labels = np.zeros(probmax.shape, dtype='int')
            for label in new_label_list:
                prob = utils.load_volume(os.path.join(upsample_indiv_label_dir, str(label) + '.nii.gz'))
                idx = prob > probmax
                labels[idx] = label
                probmax[idx] = prob[idx]
            utils.save_volume(label_list[labels], aff, h, path_result, dtype='int32')


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
    label_list, _ = utils.get_list_labels(path_label_list, labels_dir, FS_sort=FS_sort)

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
        volumes = np.zeros((label_list.shape[0] - 1, len(path_labels)))
    else:
        volumes = np.zeros((label_list.shape[0], len(path_labels)))
    loop_info = utils.LoopInfo(len(path_labels), 10, 'processing', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

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


def build_atlas(labels_dir,
                label_list,
                align_centre_of_mass=False,
                margin=15,
                shape=None,
                path_atlas=None):
    """This function builds a binary atlas (defined by label values > 0) from several label maps.
    :param labels_dir: path of directory with input label maps
    :param label_list: list of all labels in the label maps. If there is more than 1 value here, the different channels
    of the atlas (each corresponding to the probability map of a given label) will in the same order as in this list.
    :param align_centre_of_mass: whether to build the atlas by aligning the center of mass of each label map.
    If False, the atlas has the same size as the input label maps, which are assumed to be aligned.
    :param margin: (optional) If align_centre_of_mass is True, margin by which to crop the input label maps around
    their center of mass. Therefore it controls the size of the output atlas: (2*margin + 1)**n_dims.
    :param shape: shape of the output atlas.
    :param path_atlas: (optional) path where the output atlas will be writen.
    Default is None, where the atlas is not saved."""

    # list of all label maps and create result dir
    path_labels = utils.list_images_in_folder(labels_dir)
    n_label_maps = len(path_labels)
    utils.mkdir(os.path.dirname(path_atlas))

    # read list labels and create lut
    label_list = np.array(utils.reformat_to_list(label_list, load_as_numpy=True, dtype='int'))
    lut = utils.get_mapping_lut(label_list)
    n_labels = len(label_list)

    # create empty atlas
    im_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_labels[0], aff_ref=np.eye(4))
    if align_centre_of_mass:
        shape = [margin * 2] * n_dims
    else:
        shape = utils.reformat_to_list(shape, length=n_dims) if shape is not None else im_shape
    shape = shape + [n_labels] if n_labels > 1 else shape
    atlas = np.zeros(shape)

    # loop over label maps
    loop_info = utils.LoopInfo(n_label_maps, 10, 'processing', True)
    for idx, path_label in enumerate(path_labels):
        loop_info.update(idx)

        # load label map and build mask
        lab = utils.load_volume(path_label, dtype='int32', aff_ref=np.eye(4))
        lab = correct_label_map(lab, [31, 63, 72], [4, 43, 0])
        lab = lut[lab.astype('int')]
        lab = pad_volume(lab, shape[:n_dims])
        lab = crop_volume(lab, cropping_shape=shape[:n_dims])
        indices = np.where(lab > 0)

        if len(label_list) > 1:
            lab = np.identity(n_labels)[lab]

        # crop label map around centre of mass
        if align_centre_of_mass:
            centre_of_mass = np.array([np.mean(indices[0]), np.mean(indices[1]), np.mean(indices[2])], dtype='int32')
            min_crop = centre_of_mass - margin
            max_crop = centre_of_mass + margin
            atlas += lab[min_crop[0]:max_crop[0], min_crop[1]:max_crop[1], min_crop[2]:max_crop[2], ...]
        # otherwise just add the one-hot labels
        else:
            atlas += lab

    # normalise atlas and save it if necessary
    atlas /= n_label_maps
    atlas = align_volume_to_ref(atlas, np.eye(4), aff_ref=aff, n_dims=n_dims)
    if path_atlas is not None:
        utils.save_volume(atlas, aff, h, path_atlas)

    return atlas


# ---------------------------------------------------- edit dataset ----------------------------------------------------

def check_images_and_labels(image_dir, labels_dir, verbose=True):
    """Check if corresponding images and labels have the same affine matrices and shapes.
    Labels are matched to images by sorting order.
    :param image_dir: path of directory with input images
    :param labels_dir: path of directory with corresponding label maps
    :param verbose: whether to print out info
    """

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), 'different number of files in image_dir and labels_dir'

    # loop over images and labels
    loop_info = utils.LoopInfo(len(path_images), 10, 'checking', verbose) if verbose else None
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        if loop_info is not None:
            loop_info.update(idx)

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


def crop_dataset_to_minimum_size(labels_dir, result_dir, image_dir=None, image_result_dir=None, margin=5):
    """Crop all label maps in a directory to the minimum possible common size, with a margin.
    This is achieved by cropping each label map individually to the minimum size, and by padding all the cropped maps to
    the same size (taken to be the maximum size of the cropped maps).
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
    loop_info = utils.LoopInfo(len(path_labels), 10, 'cropping', True)
    for idx, (path_label, path_image) in enumerate(zip(path_labels, path_images)):
        loop_info.update(idx)

        # crop label maps and update maximum size of cropped map
        label, aff, h = utils.load_volume(path_label, im_only=False)
        label, cropping, aff = crop_volume_around_region(label, aff=aff)
        utils.save_volume(label, aff, h, os.path.join(result_dir, os.path.basename(path_label)))
        maximum_size = np.maximum(maximum_size, np.array(label.shape) + margin * 2)  # *2 to add margin on each side

        # crop images if required
        if path_image is not None:
            image, aff_im, h_im = utils.load_volume(path_image, im_only=False)
            image, aff_im = crop_volume_with_idx(image, cropping, aff=aff_im)
            utils.save_volume(image, aff_im, h_im, os.path.join(image_result_dir, os.path.basename(path_image)))

    # loop over label maps for padding
    print('\npadding labels to same size')
    loop_info = utils.LoopInfo(len(path_labels), 10, 'padding', True)
    for idx, (path_label, path_image) in enumerate(zip(path_labels, path_images)):
        loop_info.update(idx)

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


def crop_dataset_around_region_of_same_size(labels_dir,
                                            result_dir,
                                            image_dir=None,
                                            image_result_dir=None,
                                            margin=0,
                                            recompute=True):

    # create result dir
    utils.mkdir(result_dir)
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        utils.mkdir(image_result_dir)

    # list labels and images
    path_labels = utils.list_images_in_folder(labels_dir)
    path_images = utils.list_images_in_folder(image_dir) if image_dir is not None else [None] * len(path_labels)
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])

    recompute_labels = any([not os.path.isfile(os.path.join(result_dir, os.path.basename(path)))
                            for path in path_labels])
    if (image_dir is not None) & (not recompute_labels):
        recompute_labels = any([not os.path.isfile(os.path.join(image_result_dir, os.path.basename(path)))
                                for path in path_images])

    # get minimum patch shape so that no labels are left out when doing the cropping later on
    max_crop_shape = np.zeros(n_dims)
    if recompute_labels:
        for path_label in path_labels:
            label, aff, _ = utils.load_volume(path_label, im_only=False)
            label = align_volume_to_ref(label, aff, aff_ref=np.eye(4))
            label = get_largest_connected_component(label > 0, structure=np.ones((3, 3, 3)))
            _, cropping = crop_volume_around_region(label)
            max_crop_shape = np.maximum(cropping[n_dims:] - cropping[:n_dims], max_crop_shape)
        max_crop_shape += np.array(utils.reformat_to_list(margin, length=n_dims, dtype='int'))
        print('max_crop_shape: ', max_crop_shape)

    # crop shapes (possibly with padding if images are smaller than crop shape)
    for path_label, path_image in zip(path_labels, path_images):

        path_label_result = os.path.join(result_dir, os.path.basename(path_label))
        path_image_result = os.path.join(image_result_dir, os.path.basename(path_image))

        if (not os.path.isfile(path_image_result)) | (not os.path.isfile(path_label_result)) | recompute:
            # load labels
            label, aff, h_la = utils.load_volume(path_label, im_only=False, dtype='int32')
            label, aff_new = align_volume_to_ref(label, aff, aff_ref=np.eye(4), return_aff=True)
            vol_shape = np.array(label.shape[:n_dims])
            if path_image is not None:
                image, _, h_im = utils.load_volume(path_image, im_only=False)
                image = align_volume_to_ref(image, aff, aff_ref=np.eye(4))
            else:
                image = h_im = None

            # mask labels
            mask = get_largest_connected_component(label > 0, structure=np.ones((3, 3, 3)))
            label[np.logical_not(mask)] = 0

            # find cropping indices
            indices = np.nonzero(mask)
            min_idx = np.maximum(np.array([np.min(idx) for idx in indices]) - margin, 0)
            max_idx = np.minimum(np.array([np.max(idx) for idx in indices]) + 1 + margin, vol_shape)

            # expand/retract (depending on the desired shape) the cropping region around the centre
            intermediate_vol_shape = max_idx - min_idx
            min_idx = min_idx - np.int32(np.ceil((max_crop_shape - intermediate_vol_shape) / 2))
            max_idx = max_idx + np.int32(np.floor((max_crop_shape - intermediate_vol_shape) / 2))

            # check if we need to pad the output to the desired shape
            min_padding = np.abs(np.minimum(min_idx, 0))
            max_padding = np.maximum(max_idx - vol_shape, 0)
            if np.any(min_padding > 0) | np.any(max_padding > 0):
                pad_margins = tuple([(min_padding[i], max_padding[i]) for i in range(n_dims)])
            else:
                pad_margins = None
            cropping = np.concatenate([np.maximum(min_idx, 0), np.minimum(max_idx, vol_shape)])

            # crop volume
            label = crop_volume_with_idx(label, cropping, n_dims=n_dims)
            if path_image is not None:
                image = crop_volume_with_idx(image, cropping, n_dims=n_dims)

            # pad volume if necessary
            if pad_margins is not None:
                label = np.pad(label, pad_margins, mode='constant', constant_values=0)
                if path_image is not None:
                    _, n_channels = utils.get_dims(image.shape)
                    pad_margins = tuple(list(pad_margins) + [(0, 0)]) if n_channels > 1 else pad_margins
                    image = np.pad(image, pad_margins, mode='constant', constant_values=0)

            # update aff
            if n_dims == 2:
                min_idx = np.append(min_idx, 0)
            aff_new[0:3, -1] = aff_new[0:3, -1] + aff_new[:3, :3] @ min_idx

            # write labels
            label, aff_final = align_volume_to_ref(label, aff_new, aff_ref=aff, return_aff=True)
            utils.save_volume(label, aff_final, h_la, path_label_result, dtype='int32')
            if path_image is not None:
                image = align_volume_to_ref(image, aff_new, aff_ref=aff)
                utils.save_volume(image, aff_final, h_im, path_image_result)


def crop_dataset_around_region(image_dir, labels_dir, image_result_dir, labels_result_dir, margin=0,
                               cropping_shape_div_by=None, recompute=True):

    # create result dir
    utils.mkdir(image_result_dir)
    utils.mkdir(labels_result_dir)

    # list volumes and masks
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_labels[0])

    # loop over images and labels
    loop_info = utils.LoopInfo(len(path_images), 10, 'cropping', True)
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        loop_info.update(idx)

        path_label_result = os.path.join(labels_result_dir, os.path.basename(path_label))
        path_image_result = os.path.join(image_result_dir, os.path.basename(path_image))

        if (not os.path.isfile(path_label_result)) | (not os.path.isfile(path_image_result)) | recompute:

            image, aff, h_im = utils.load_volume(path_image, im_only=False)
            label, _, h_lab = utils.load_volume(path_label, im_only=False)
            mask = get_largest_connected_component(label > 0, structure=np.ones((3, 3, 3)))
            label[np.logical_not(mask)] = 0
            vol_shape = np.array(label.shape[:n_dims])

            # find cropping indices
            indices = np.nonzero(mask)
            min_idx = np.maximum(np.array([np.min(idx) for idx in indices]) - margin, 0)
            max_idx = np.minimum(np.array([np.max(idx) for idx in indices]) + 1 + margin, vol_shape)

            # expand/retract (depending on the desired shape) the cropping region around the centre
            intermediate_vol_shape = max_idx - min_idx
            cropping_shape = np.array([utils.find_closest_number_divisible_by_m(s, cropping_shape_div_by,
                                                                                answer_type='higher')
                                       for s in intermediate_vol_shape])
            min_idx = min_idx - np.int32(np.ceil((cropping_shape - intermediate_vol_shape) / 2))
            max_idx = max_idx + np.int32(np.floor((cropping_shape - intermediate_vol_shape) / 2))

            # check if we need to pad the output to the desired shape
            min_padding = np.abs(np.minimum(min_idx, 0))
            max_padding = np.maximum(max_idx - vol_shape, 0)
            if np.any(min_padding > 0) | np.any(max_padding > 0):
                pad_margins = tuple([(min_padding[i], max_padding[i]) for i in range(n_dims)])
            else:
                pad_margins = None
            cropping = np.concatenate([np.maximum(min_idx, 0), np.minimum(max_idx, vol_shape)])

            # crop volume
            label = crop_volume_with_idx(label, cropping, n_dims=n_dims)
            image = crop_volume_with_idx(image, cropping, n_dims=n_dims)

            # pad volume if necessary
            if pad_margins is not None:
                label = np.pad(label, pad_margins, mode='constant', constant_values=0)
                pad_margins = tuple(list(pad_margins) + [(0, 0)]) if n_channels > 1 else pad_margins
                image = np.pad(image, pad_margins, mode='constant', constant_values=0)

            # update aff
            if n_dims == 2:
                min_idx = np.append(min_idx, 0)
            aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ min_idx

            # write results
            utils.save_volume(image, aff, h_im, path_image_result)
            utils.save_volume(label, aff, h_lab, path_label_result, dtype='int32')


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
    loop_info = utils.LoopInfo(len(path_images), 10, 'processing', True)
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        loop_info.update(idx)

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
                        temp_la = lab[i:i + patch_shape[0], j:j + patch_shape[1], ...]
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
                            temp_la = lab[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2], ...]
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
