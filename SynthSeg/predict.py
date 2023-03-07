"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

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
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# project imports
from SynthSeg import evaluate

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models


def predict(path_images,
            path_segmentations,
            path_model,
            labels_segmentation,
            n_neutral_labels=None,
            names_segmentation=None,
            path_posteriors=None,
            path_resampled=None,
            path_volumes=None,
            min_pad=None,
            cropping=None,
            target_res=1.,
            gradients=False,
            flip=True,
            topology_classes=None,
            sigma_smoothing=0.5,
            keep_biggest_component=True,
            n_levels=5,
            nb_conv_per_level=2,
            conv_size=3,
            unet_feat_count=24,
            feat_multiplier=2,
            activation='elu',
            gt_folder=None,
            evaluation_labels=None,
            list_incorrect_labels=None,
            list_correct_labels=None,
            compute_distances=False,
            recompute=True,
            verbose=True):
    """
    This function uses trained models to segment images.
    It is crucial that the inputs match the architecture parameters of the trained model.
    :param path_images: path of the images to segment. Can be the path to a directory or the path to a single image.
    :param path_segmentations: path where segmentations will be writen.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    :param path_model: path ot the trained model.
    :param labels_segmentation: List of labels for which to compute Dice scores. It should be the same list as the
    segmentation_labels used in training.
    :param n_neutral_labels: (optional) if the label maps contain some right/left specific labels and if test-time
    flipping is applied (see parameter 'flip'), please provide the number of non-sided labels (including background).
    It should be the same value as for training. Default is None.
    :param names_segmentation: (optional) List of names corresponding to the names of the segmentation labels.
    Only used when path_volumes is provided. Must be of the same size as segmentation_labels. Can be given as a
    list, a numpy array of strings, or the path to such a numpy array. Default is None.
    :param path_posteriors: (optional) path where posteriors will be writen.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    :param path_resampled: (optional) path where images resampled to 1mm isotropic will be writen.
    We emphasise that images are resampled as soon as the resolution in one of the axes is not in the range [0.9; 1.1].
    Should be a dir, if path_images is a dir, and a file if path_images is a file. Default is None, where resampled
    images are not saved.
    :param path_volumes: (optional) path of a csv file where the soft volumes of all segmented regions will be writen.
    The rows of the csv file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    :param min_pad: (optional) minimum size of the images to process. Can be an int, a sequence or a 1d numpy array.
    :param cropping: (optional) crop the images to the specified shape before predicting the segmentation maps.
    Cropping overwrites min_pad if min_pad>cropping. Can be an int, a sequence or a 1d numpy array.
    :param target_res: (optional) target resolution at which the network operates (and thus resolution of the output
    segmentations). This must match the resolution of the training data ! target_res is used to automatically resampled
    the images with resolutions outside [target_res-0.05, target_res+0.05].
    :param gradients: (optional) whether to replace the image by the magnitude of its gradient as input to the network.
    Can be a sequence, a 1d numpy array. Set to None to disable the automatic resampling. Default is 1mm.
    :param flip: (optional) whether to perform test-time augmentation, where the input image is segmented along with
    a right/left flipped version on it. If set to True (default), be careful because this requires more memory.
    :param topology_classes: List of classes corresponding to all segmentation labels, in order to group them into
    classes, for each of which we will operate a smooth version of biggest connected component.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array in the same order as segmentation_labels.
    Default is None, where no topological analysis is performed.
    :param sigma_smoothing: (optional) If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    :param keep_biggest_component: (optional) whether to only keep the biggest component in the predicted segmentation.
    This is applied independently of topology_classes, and it is applied to the whole segmentation
    :param n_levels: (optional) number of levels for unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolution layers per level. Default is 2.
    :param conv_size: (optional) size of UNet's convolution masks. Default is 3.
    :param unet_feat_count: (optional) number of features for the first layer of the unet. Default is 24.
    :param feat_multiplier: (optional) multiplicative factor for the number of feature for each new level. Default is 2.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.
    :param gt_folder: (optional) path of the ground truth label maps corresponding to the input images. Should be a dir,
    if path_images is a dir, or a file if path_images is a file.
    Providing a gt_folder will trigger a Dice evaluation, where scores will be writen along with the path_segmentations.
    Specifically, the scores are contained in a numpy array, where labels are in rows, and subjects in columns.
    :param evaluation_labels: (optional) if gt_folder is True you can evaluate the Dice scores on a subset of the
    segmentation labels, by providing another label list here. Can be a sequence, a 1d numpy array, or the path to a
    numpy 1d array. Default is np.unique(segmentation_labels).
    :param list_incorrect_labels: (optional) this option enables to replace some label values in the obtained
    segmentations by other label values. Can be a list, a 1d numpy array, or the path to such an array.
    :param list_correct_labels: (optional) list of values to correct the labels specified in list_incorrect_labels.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    :param compute_distances: (optional) whether to add Hausdorff and mean surface distance evaluations to the default
    Dice evaluation. Default is True.
    :param recompute: (optional) whether to recompute segmentations that were already computed. This also applies to
    Dice scores, if gt_folder is not None. Default is True.
    :param verbose: (optional) whether to print out info about the remaining number of cases.
    """

    # prepare input/output filepaths
    path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, compute, unique_vol_file = \
        prepare_output_files(path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, recompute)

    # get label list
    labels_segmentation, _ = utils.get_list_labels(label_list=labels_segmentation)
    if (n_neutral_labels is not None) & flip:
        labels_segmentation, flip_indices, unique_idx = get_flip_indices(labels_segmentation, n_neutral_labels)
    else:
        labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
        flip_indices = None

    # prepare other labels list
    if names_segmentation is not None:
        names_segmentation = utils.load_array_if_path(names_segmentation)[unique_idx]
    if topology_classes is not None:
        topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]

    # prepare volumes if necessary
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, labels_segmentation, names_segmentation)

    # build network
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_images[0])
    model_input_shape = [None] * n_dims + [n_channels]
    net = build_model(path_model=path_model,
                      input_shape=model_input_shape,
                      labels_segmentation=labels_segmentation,
                      n_levels=n_levels,
                      nb_conv_per_level=nb_conv_per_level,
                      conv_size=conv_size,
                      unet_feat_count=unet_feat_count,
                      feat_multiplier=feat_multiplier,
                      activation=activation,
                      sigma_smoothing=sigma_smoothing,
                      flip_indices=flip_indices,
                      gradients=gradients)

    # set cropping/padding
    if (cropping is not None) & (min_pad is not None):
        cropping = utils.reformat_to_list(cropping, length=n_dims, dtype='int')
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = np.minimum(cropping, min_pad)

    # perform segmentation
    if len(path_images) <= 10:
        loop_info = utils.LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = utils.LoopInfo(len(path_images), 10, 'predicting', True)
    for i in range(len(path_images)):
        if verbose:
            loop_info.update(i)

        # compute segmentation only if needed
        if compute[i]:

            # preprocessing
            image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image=path_images[i],
                                                                         n_levels=n_levels,
                                                                         target_res=target_res,
                                                                         crop=cropping,
                                                                         min_pad=min_pad,
                                                                         path_resample=path_resampled[i])

            # prediction
            post_patch = net.predict(image)

            # postprocessing
            seg, posteriors, volumes = postprocess(post_patch=post_patch,
                                                   shape=shape,
                                                   pad_idx=pad_idx,
                                                   crop_idx=crop_idx,
                                                   n_dims=n_dims,
                                                   labels_segmentation=labels_segmentation,
                                                   keep_biggest_component=keep_biggest_component,
                                                   aff=aff,
                                                   im_res=im_res,
                                                   topology_classes=topology_classes)

            # write results to disk
            utils.save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
            if path_posteriors[i] is not None:
                if n_channels > 1:
                    posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')

            # compute volumes
            if path_volumes[i] is not None:
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, labels_segmentation, names_segmentation)

    # evaluate
    if gt_folder is not None:

        # find path where segmentations are saved evaluation folder, and get labels on which to evaluate
        eval_folder = os.path.dirname(path_segmentations[0])
        if evaluation_labels is None:
            evaluation_labels = labels_segmentation

        # set path of result arrays for surface distance if necessary
        if compute_distances:
            path_hausdorff = os.path.join(eval_folder, 'hausdorff.npy')
            path_hausdorff_99 = os.path.join(eval_folder, 'hausdorff_99.npy')
            path_hausdorff_95 = os.path.join(eval_folder, 'hausdorff_95.npy')
            path_mean_distance = os.path.join(eval_folder, 'mean_distance.npy')
        else:
            path_hausdorff = path_hausdorff_99 = path_hausdorff_95 = path_mean_distance = None

        # compute evaluation metrics
        evaluate.evaluation(gt_folder,
                            eval_folder,
                            evaluation_labels,
                            path_dice=os.path.join(eval_folder, 'dice.npy'),
                            path_hausdorff=path_hausdorff,
                            path_hausdorff_99=path_hausdorff_99,
                            path_hausdorff_95=path_hausdorff_95,
                            path_mean_distance=path_mean_distance,
                            list_incorrect_labels=list_incorrect_labels,
                            list_correct_labels=list_correct_labels,
                            recompute=recompute,
                            verbose=verbose)


def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute):

    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert out_seg is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a text file
    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # define helper to deal with outputs
        def text_helper(path, name):
            if path is not None:
                assert path[-4:] == '.txt', 'if path_images given as text file, so must be %s' % name
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            unique_file = False
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = text_helper(out_seg, 'path_segmentations')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_resampled, recompute_resampled, _ = text_helper(out_resampled, 'path_resampled')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input images
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

        # define helper to deal with outputs
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (path[-4:] == '.npz'):
                        raise Exception('Output FOLDER had a FILE extension' % path)
                    path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                    path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                    path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                    path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                utils.mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_dir(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_dir(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')

    # path_images is an image
    else:

        # input image
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

        # define helper to deal with outputs
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                        file_name = os.path.basename(path_images[0]).replace('.nii', '_%s.nii' % suffix)
                        file_name = file_name.replace('.mgz', '_%s.mgz' % suffix)
                        file_name = file_name.replace('.npz', '_%s.npz' % suffix)
                        path = os.path.join(path, file_name)
                    recompute_files = [not os.path.isfile(path)]
                utils.mkdir(os.path.dirname(path))
            else:
                recompute_files = [False]
            path = [path]
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, recompute_resampled, _ = helper_im(out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    recompute_list = [recompute | re_seg | re_post | re_res | re_vol for (re_seg, re_post, re_res, re_vol)
                      in zip(recompute_seg, recompute_post, recompute_resampled, recompute_volume)]

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute_list, unique_volume_file


def preprocess(path_image, n_levels, target_res, crop=None, min_pad=None, path_resample=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)

    # resample image if necessary
    if target_res is not None:
        target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
        if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
            im_res = target_res
            im, aff = edit_volumes.resample_volume(im, aff, im_res)
            if path_resample is not None:
                utils.save_volume(im, aff, h, path_resample)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    if n_channels == 1:
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)
    else:
        for i in range(im.shape[-1]):
            im[..., i] = edit_volumes.rescale_volume(im[..., i], new_min=0., new_max=1.,
                                                     min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:  # in SynthSeg predict use crop flag and then if used do min_pad=crop else min_pad = 192
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx


def build_model(path_model,
                input_shape,
                labels_segmentation,
                n_levels,
                nb_conv_per_level,
                conv_size,
                unet_feat_count,
                feat_multiplier,
                activation,
                sigma_smoothing,
                flip_indices,
                gradients):

    assert os.path.isfile(path_model), "The provided model path does not exist."

    # get labels
    n_labels_seg = len(labels_segmentation)

    if gradients:
        input_image = KL.Input(input_shape)
        last_tensor = layers.ImageGradients('sobel', True)(input_image)
        last_tensor = KL.Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + K.epsilon()))(last_tensor)
        net = Model(inputs=input_image, outputs=last_tensor)
    else:
        net = None

    # build UNet
    net = nrn_models.unet(input_model=net,
                          input_shape=input_shape,
                          nb_labels=n_labels_seg,
                          nb_levels=n_levels,
                          nb_conv_per_level=nb_conv_per_level,
                          conv_size=conv_size,
                          nb_features=unet_feat_count,
                          feat_mult=feat_multiplier,
                          activation=activation,
                          batch_norm=-1)
    net.load_weights(path_model, by_name=True)

    # smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

    if flip_indices is not None:

        # segment flipped image
        input_image = net.inputs[0]
        seg = net.output
        image_flipped = layers.RandomFlip(axis=0, prob=1)(input_image)
        last_tensor = net(image_flipped)

        # flip back and re-order channels
        last_tensor = layers.RandomFlip(axis=0, prob=1)(last_tensor)
        last_tensor = KL.Lambda(lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
        reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
        last_tensor = KL.Lambda(lambda x: tf.concat(x, -1), name='concat')(reordered_channels)

        # average two segmentations and build model
        name_segm_prediction_layer = 'average_lr'
        last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
        net = Model(inputs=net.inputs, outputs=last_tensor)

    return net


def postprocess(post_patch, shape, pad_idx, crop_idx, n_dims,
                labels_segmentation, keep_biggest_component, aff, im_res, topology_classes=None):

    # get posteriors
    post_patch = np.squeeze(post_patch)
    if topology_classes is None:
        post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)

    # keep biggest connected component
    if keep_biggest_component:
        tmp_post_patch = post_patch[..., 1:]
        post_patch_mask = np.sum(tmp_post_patch, axis=-1) > 0.25
        post_patch_mask = edit_volumes.get_largest_connected_component(post_patch_mask)
        post_patch_mask = np.stack([post_patch_mask]*tmp_post_patch.shape[-1], axis=-1)
        tmp_post_patch = edit_volumes.mask_volume(tmp_post_patch, mask=post_patch_mask, return_copy=False)
        post_patch[..., 1:] = tmp_post_patch

    # reset posteriors to zero outside the largest connected component of each topological class
    if topology_classes is not None:
        post_patch_mask = post_patch > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = edit_volumes.get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch[..., idx] *= tmp_mask
        post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)

    # normalise posteriors and get hard segmentation
    if keep_biggest_component | (topology_classes is not None):
        post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch.argmax(-1).astype('int32')].astype('int32')

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        posteriors[..., 0] = np.ones(shape)  # place background around patch
        if n_dims == 2:
            seg[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = seg_patch
            posteriors[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], :] = post_patch
        elif n_dims == 3:
            seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
            posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = post_patch
    else:
        seg = seg_patch
        posteriors = post_patch

    # align prediction back to first orientation
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)
    posteriors = edit_volumes.align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)

    # compute volumes
    volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
    volumes = np.around(volumes * np.prod(im_res), 3)

    return seg, posteriors, volumes


def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondence between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]

    return labels_segmentation, flip_indices, unique_idx


def write_csv(path_csv, data, unique_file, labels, names, skip_first=True, last_first=False):

    # initialisation
    utils.mkdir(os.path.dirname(path_csv))
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names = names[unique_idx].tolist()
        if skip_first:
            names = names[1:]
        header = names
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
