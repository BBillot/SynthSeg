"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import os
import csv
import numpy as np
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
            segmentation_labels,
            n_neutral_labels=None,
            path_posteriors=None,
            path_resampled=None,
            path_volumes=None,
            segmentation_label_names=None,
            min_pad=None,
            cropping=None,
            target_res=1.,
            gradients=False,
            flip=True,
            topology_classes=None,
            sigma_smoothing=0.5,
            keep_biggest_component=True,
            conv_size=3,
            n_levels=5,
            nb_conv_per_level=2,
            unet_feat_count=24,
            feat_multiplier=2,
            activation='elu',
            gt_folder=None,
            evaluation_labels=None,
            mask_folder=None,
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
    :param segmentation_labels: List of labels for which to compute Dice scores. It should be the same list as the
    segmentation_labels used in training.
    :param n_neutral_labels: (optional) if the label maps contain some right/left specific labels and if test-time
    flipping is applied (see parameter 'flip'), please provide the number of non-sided labels (including background).
    It should be the same value as for training. Default is None.
    :param path_posteriors: (optional) path where posteriors will be writen.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    :param path_resampled: (optional) path where images resampled to 1mm isotropic will be writen.
    We emphasise that images are resampled as soon as the resolution in one of the axes is not in the range [0.9; 1.1].
    Should be a dir, if path_images is a dir, and a file if path_images is a file. Default is None, where resampled
    images are not saved.
    :param path_volumes: (optional) path of a csv file where the soft volumes of all segmented regions will be writen.
    The rows of the csv file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    :param segmentation_label_names: (optional) List of names corresponding to the names of the segmentation labels.
    Only used when path_volumes is provided. Must be of the same size as segmentation_labels. Can be given as a
    list, a numpy array of strings, or the path to such a numpy array. Default is None.
    :param min_pad: (optional) minimum size of the images to process. Can be an int, a sequence or a 1d numpy array.
    :param cropping: (optional) crop the images to the specified shape before predicting the segmentation maps.
    Cropping overwrites min_pad if min_pad>cropping. Can be an int, a sequence or a 1d numpy array.
    :param target_res: (optional) target resolution at which the network operates (and thus resolution of the output
    segmentations). This must match the resolution of the training data ! target_res is used to automatically resampled
    the images with resolutions outside [target_res-0.05, target_res+0.05].
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
    :param conv_size: (optional) size of UNet's convolution masks. Default is 3.
    :param n_levels: (optional) number of levels for unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolution layers per level. Default is 2.
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
    :param mask_folder: (optional) path of masks that will be used to mask out some parts of the obtained segmentations
    during the evaluation. Default is None, where nothing is masked.
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
    segmentation_labels, _ = utils.get_list_labels(label_list=segmentation_labels)
    n_labels = len(segmentation_labels)

    # get unique label values, and build correspondence table between contralateral structures if necessary
    if (n_neutral_labels is not None) & flip:
        n_sided_labels = int((n_labels - n_neutral_labels) / 2)
        lr_corresp = np.stack([segmentation_labels[n_neutral_labels:n_neutral_labels + n_sided_labels],
                               segmentation_labels[n_neutral_labels + n_sided_labels:]])
        segmentation_labels, indices = np.unique(segmentation_labels, return_index=True)
        lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
        lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
        lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique
        lr_indices = np.zeros_like(lr_corresp_unique)
        for i in range(lr_corresp_unique.shape[0]):
            for j, lab in enumerate(lr_corresp_unique[i]):
                lr_indices[i, j] = np.where(segmentation_labels == lab)[0]
    else:
        segmentation_labels, indices = np.unique(segmentation_labels, return_index=True)
        lr_indices = None

    # prepare topology classes
    if topology_classes is not None:
        topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[indices]

    # prepare volume file if needed
    if unique_vol_file & (path_volumes[0] is not None):
        if segmentation_label_names is not None:
            segmentation_label_names = utils.load_array_if_path(segmentation_label_names)[indices]
            csv_header = [[''] + segmentation_label_names[1:].tolist()]
            csv_header += [[''] + [str(lab) for lab in segmentation_labels[1:]]]
        else:
            csv_header = [['subjects'] + [str(lab) for lab in segmentation_labels[1:]]]
        with open(path_volumes[0], 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csv_header)

    # build network
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_images[0])
    model_input_shape = [None] * n_dims + [n_channels]
    net = build_model(path_model, model_input_shape, n_levels, len(segmentation_labels), conv_size,
                      nb_conv_per_level, unet_feat_count, feat_multiplier, activation, sigma_smoothing, gradients)

    if (cropping is not None) & (min_pad is not None):
        cropping = utils.reformat_to_list(cropping, length=n_dims, dtype='int')
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = np.minimum(cropping, min_pad)

    # perform segmentation
    loop_info = utils.LoopInfo(len(path_images), 10, 'predicting', True)
    for idx, (path_image, path_segmentation, path_posterior, path_resample, path_volume, tmp_compute) in \
            enumerate(zip(path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, compute)):

        # compute segmentation only if needed
        if tmp_compute:
            if verbose:
                loop_info.update(idx)

            # preprocessing
            image, aff, h, im_res, shape, pad_idx, crop_idx, im_flipped = \
                preprocess(path_image, n_levels, target_res, cropping, min_pad, flip, path_resample)

            # prediction
            prediction_patch = net.predict(image)
            prediction_patch_flip = net.predict(im_flipped) if flip else None

            # postprocessing
            seg, posteriors = postprocess(prediction_patch, shape, pad_idx, crop_idx, n_dims, segmentation_labels,
                                          lr_indices, keep_biggest_component, aff,
                                          topology_classes=topology_classes, post_patch_flip=prediction_patch_flip)

            # write results to disk
            utils.save_volume(seg, aff, h, path_segmentation, dtype='int32')
            if path_posterior is not None:
                if n_channels > 1:
                    posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors, aff, h, path_posterior, dtype='float32')

            # compute volumes
            if path_volume is not None:
                utils.mkdir(os.path.dirname(path_volume))
                volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
                volumes = np.around(volumes * np.prod(im_res), 3)
                row = [os.path.basename(path_image).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                if unique_vol_file:
                    with open(path_volume, 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(row)
                else:
                    rows = [[''] + [str(lab) for lab in segmentation_labels[1:]], row]
                    with open(path_volume, 'w') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerows(rows)

    # evaluate
    if gt_folder is not None:

        # find path where segmentations are saved evaluation folder, and get labels on which to evaluate
        eval_folder = os.path.dirname(path_segmentations[0])
        if evaluation_labels is None:
            evaluation_labels = segmentation_labels

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
                            mask_dir=mask_folder,
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
    '''
    Prepare output files.
    '''

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

    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # segmentations
        assert out_seg[-4:] == '.txt', 'if path_images given as text file, so must be the output segmentations'
        with open(out_seg, 'r') as f:
            out_seg = [line.replace('\n', '') for line in f.readlines() if line != '\n']
        recompute_seg = [not os.path.isfile(path_seg) for path_seg in out_seg]

        # volumes
        if out_volumes is not None:
            assert out_volumes[-4:] == '.txt', 'if path_images given as text file, so must be the output volumes'
            with open(out_volumes, 'r') as f:
                out_volumes = [line.replace('\n', '') for line in f.readlines() if line != '\n']
            recompute_volume = [not os.path.isfile(path_vol) for path_vol in out_volumes]
        else:
            out_volumes = [None] * len(path_images)
            recompute_volume = [False] * len(path_images)
        unique_volume_file = False

        # posteriors
        if out_posteriors is not None:
            assert out_posteriors[-4:] == '.txt', 'if path_images given as text file, so must be the output posteriors'
            with open(out_posteriors, 'r') as f:
                out_posteriors = [line.replace('\n', '') for line in f.readlines() if line != '\n']
            recompute_post = [not os.path.isfile(path_post) for path_post in out_posteriors]
        else:
            out_posteriors = [None] * len(path_images)
            recompute_post = [False] * len(path_images)

        # resampled
        if out_resampled is not None:
            assert out_resampled[-4:] == '.txt', 'if path_images given as text file, so must be the resampled images'
            with open(out_resampled, 'r') as f:
                out_resampled = [line.replace('\n', '') for line in f.readlines() if line != '\n']
            recompute_resampled = [not os.path.isfile(path_post) for path_post in out_resampled]
        else:
            out_resampled = [None] * len(path_images)
            recompute_resampled = [False] * len(path_images)

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input images
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

        # segmentations
        assert out_seg[-4:] != '.txt', 'path_segmentations can only be given as text file when path_images is.'
        if (out_seg[-7:] == '.nii.gz') | (out_seg[-4:] == '.nii') | (out_seg[-4:] == '.mgz') | (out_seg[-4:] == '.npz'):
            raise Exception('Output folders cannot have extensions: .nii.gz, .nii, .mgz, or .npz, had %s' % out_seg)
        utils.mkdir(out_seg)
        out_seg = [os.path.join(out_seg, os.path.basename(image)).replace('.nii', '_synthseg.nii') for image in
                   path_images]
        out_seg = [seg_path.replace('.mgz', '_synthseg.mgz') for seg_path in out_seg]
        out_seg = [seg_path.replace('.npz', '_synthseg.npz') for seg_path in out_seg]
        recompute_seg = [not os.path.isfile(path_seg) for path_seg in out_seg]

        # volumes
        if out_volumes is not None:
            assert out_volumes[-4:] != '.txt', 'path_volumes can only be given as text file when path_images is.'
            if out_volumes[-4:] != '.csv':
                print('Path for volume outputs provided without csv extension. Adding csv extension.')
                out_volumes += '.csv'
            utils.mkdir(os.path.dirname(out_volumes))
            recompute_volume = [True] * len(path_images)
        else:
            recompute_volume = [False] * len(path_images)
        out_volumes = [out_volumes] * len(path_images)
        unique_volume_file = True

        # posteriors
        if out_posteriors is not None:
            assert out_posteriors[-4:] != '.txt', 'path_posteriors can only be given as text file when path_images is.'
            if (out_posteriors[-7:] == '.nii.gz') | (out_posteriors[-4:] == '.nii') | \
                    (out_posteriors[-4:] == '.mgz') | (out_posteriors[-4:] == '.npz'):
                raise Exception('Output folders cannot have extensions: '
                                '.nii.gz, .nii, .mgz, or .npz, had %s' % out_posteriors)
            utils.mkdir(out_posteriors)
            out_posteriors = [os.path.join(out_posteriors, os.path.basename(image)).replace('.nii',
                              '_posteriors.nii') for image in path_images]
            out_posteriors = [posteriors_path.replace('.mgz', '_posteriors.mgz') for posteriors_path in out_posteriors]
            out_posteriors = [posteriors_path.replace('.npz', '_posteriors.npz') for posteriors_path in out_posteriors]
            recompute_post = [not os.path.isfile(path_post) for path_post in out_posteriors]
        else:
            out_posteriors = [None] * len(path_images)
            recompute_post = [False] * len(path_images)

        # resampled
        if out_resampled is not None:
            assert out_resampled[-4:] != '.txt', 'path_resampled can only be given as text file when path_images is.'
            if (out_resampled[-7:] == '.nii.gz') | (out_resampled[-4:] == '.nii') | \
                    (out_resampled[-4:] == '.mgz') | (out_resampled[-4:] == '.npz'):
                raise Exception('Output folders cannot have extensions: '
                                '.nii.gz, .nii, .mgz, or .npz, had %s' % out_resampled)
            utils.mkdir(out_resampled)
            out_resampled = [os.path.join(out_resampled, os.path.basename(image)).replace('.nii',
                             '_resampled.nii') for image in path_images]
            out_resampled = [resampled_path.replace('.mgz', '_resampled.mgz') for resampled_path in out_resampled]
            out_resampled = [resampled_path.replace('.npz', '_resampled.npz') for resampled_path in out_resampled]
            recompute_resampled = [not os.path.isfile(path_post) for path_post in out_resampled]
        else:
            out_resampled = [None] * len(path_images)
            recompute_resampled = [False] * len(path_images)

    # path_images is an image
    else:

        # input images
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

        # segmentations
        assert out_seg[-4:] != '.txt', 'path_segmentations can only be given as text file when path_images is.'
        if ('.nii.gz' not in out_seg) & ('.nii' not in out_seg) & ('.mgz' not in out_seg) & ('.npz' not in out_seg):
            utils.mkdir(out_seg)
            filename = os.path.basename(path_images[0]).replace('.nii', '_synthseg.nii')
            filename = filename.replace('.mgz', '_synthseg.mgz')
            filename = filename.replace('.npz', '_synthseg.npz')
            out_seg = os.path.join(out_seg, filename)
        else:
            utils.mkdir(os.path.dirname(out_seg))
        recompute_seg = [not os.path.isfile(out_seg)]
        out_seg = [out_seg]

        # volumes
        if out_volumes is not None:
            assert out_volumes[-4:] != '.txt', 'path_volumes can only be given as text file when path_images is.'
            if out_volumes[-4:] != '.csv':
                print('Path for volume outputs provided without csv extension. Adding csv extension.')
                out_volumes += '.csv'
            utils.mkdir(os.path.dirname(out_volumes))
            recompute_volume = [True]
        else:
            recompute_volume = [False]
        out_volumes = [out_volumes]
        unique_volume_file = True

        # posteriors
        if out_posteriors is not None:
            assert out_posteriors[-4:] != '.txt', 'path_posteriors can only be given as text file when path_images is.'
            if ('.nii.gz' not in out_posteriors) & ('.nii' not in out_posteriors) &\
                    ('.mgz' not in out_posteriors) & ('.npz' not in out_posteriors):
                utils.mkdir(out_posteriors)
                filename = os.path.basename(path_images[0]).replace('.nii', '_posteriors.nii')
                filename = filename.replace('.mgz', '_posteriors.mgz')
                filename = filename.replace('.npz', '_posteriors.npz')
                out_posteriors = os.path.join(out_posteriors, filename)
            else:
                utils.mkdir(os.path.dirname(out_posteriors))
            recompute_post = [not os.path.isfile(out_posteriors)]
        else:
            recompute_post = [False]
        out_posteriors = [out_posteriors]

        # resampled
        if out_resampled is not None:
            assert out_resampled[-4:] != '.txt', 'path_resampled can only be given as text file when path_images is.'
            if ('.nii.gz' not in out_resampled) & ('.nii' not in out_resampled) &\
                    ('.mgz' not in out_resampled) & ('.npz' not in out_resampled):
                utils.mkdir(out_resampled)
                filename = os.path.basename(path_images[0]).replace('.nii', '_resampled.nii')
                filename = filename.replace('.mgz', '_resampled.mgz')
                filename = filename.replace('.npz', '_resampled.npz')
                out_resampled = os.path.join(out_resampled, filename)
            else:
                utils.mkdir(os.path.dirname(out_resampled))
            recompute_resampled = [not os.path.isfile(out_resampled)]
        else:
            recompute_resampled = [False]
        out_resampled = [out_resampled]

    recompute_list = [recompute | re_seg | re_post | re_res | re_vol
                      for (re_seg, re_post, re_res, re_vol)
                      in zip(recompute_seg, recompute_post, recompute_resampled, recompute_volume)]

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute_list, unique_volume_file


def preprocess(path_image, n_levels, target_res, crop=None, min_pad=None, flip=False, path_resample=None):

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

    # flip image along right/left axis
    if flip & (n_dims > 2):
        im_flipped = edit_volumes.flip_volume(im, direction='rl', aff=np.eye(4))
        im_flipped = utils.add_axis(im_flipped) if n_channels > 1 else utils.add_axis(im_flipped, axis=[0, -1])
    else:
        im_flipped = None

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx, im_flipped


def build_model(model_file, input_shape, n_levels, n_lab, conv_size, nb_conv_per_level, unet_feat_count,
                feat_multiplier, activation, sigma_smoothing, gradients):

    assert os.path.isfile(model_file), "The provided model path does not exist."

    if gradients:
        input_image = KL.Input(input_shape)
        last_tensor = layers.ImageGradients('sobel', True)(input_image)
        last_tensor = KL.Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + K.epsilon()))(last_tensor)
        net = Model(inputs=input_image, outputs=last_tensor)
    else:
        net = None

    # build UNet
    net = nrn_models.unet(nb_features=unet_feat_count,
                          input_shape=input_shape,
                          nb_levels=n_levels,
                          conv_size=conv_size,
                          nb_labels=n_lab,
                          feat_mult=feat_multiplier,
                          activation=activation,
                          nb_conv_per_level=nb_conv_per_level,
                          batch_norm=-1,
                          input_model=net)
    net.load_weights(model_file, by_name=True)

    # smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

    return net


def postprocess(post_patch, shape, pad_idx, crop_idx, n_dims, segmentation_labels, lr_indices,
                keep_biggest_component, aff, topology_classes=True, post_patch_flip=None):

    # get posteriors
    post_patch = np.squeeze(post_patch)
    if post_patch_flip is not None:
        post_patch_flip = np.squeeze(post_patch_flip)
        post_patch_flip = edit_volumes.flip_volume(post_patch_flip, direction='rl', aff=np.eye(4), return_copy=False)
        if lr_indices is not None:
            post_patch_flip[..., lr_indices.flatten()] = post_patch_flip[..., lr_indices[::-1].flatten()]
        post_patch = 0.5 * (post_patch + post_patch_flip)

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

    # renormalise posteriors and get hard segmentation
    if (post_patch_flip is not None) | keep_biggest_component | (topology_classes is not None):
        post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = post_patch.argmax(-1)

    # paste patches back to matrix of original image size
    seg_patch = edit_volumes.crop_volume_with_idx(seg_patch, pad_idx, n_dims=n_dims, return_copy=False)
    post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=n_dims, return_copy=False)
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, segmentation_labels.shape[0]])
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
    seg = segmentation_labels[seg.astype('int')].astype('int')

    # align prediction back to first orientation
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)
    posteriors = edit_volumes.align_volume_to_ref(posteriors, np.eye(4), aff_ref=aff, n_dims=n_dims, return_copy=False)

    return seg, posteriors
