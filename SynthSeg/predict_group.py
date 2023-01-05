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
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# project imports
from SynthSeg import evaluate
from SynthSeg.predict import write_csv

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models


def predict(path_images,
            path_masks,
            path_segmentations,
            path_model,
            labels_segmentation,
            labels_mask,
            path_posteriors=None,
            path_volumes=None,
            names_segmentation=None,
            min_pad=None,
            cropping=None,
            sigma_smoothing=0.5,
            strict_masking=False,
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

    # prepare input/output filepaths
    path_images, path_masks, path_segmentations, path_posteriors, path_volumes, compute, unique_vol_file = \
        prepare_output_files(path_images, path_masks, path_segmentations, path_posteriors, path_volumes, recompute)

    # get label list
    labels_mask, _ = utils.get_list_labels(label_list=labels_mask)
    mask_labels_unique = np.unique(labels_mask)
    labels_segmentation, _ = utils.get_list_labels(label_list=labels_segmentation)
    labels_segmentation, indices = np.unique(labels_segmentation, return_index=True)

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
                      mask_labels=mask_labels_unique)

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
            image, mask, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image=path_images[i],
                                                                               path_mask=path_masks[i],
                                                                               n_levels=n_levels,
                                                                               crop=cropping,
                                                                               min_pad=min_pad)

            # prediction
            post_patch = net.predict([image, mask])

            # postprocessing
            seg, posteriors, volumes = postprocess(post_patch=post_patch,
                                                   mask=mask,
                                                   shape=shape,
                                                   pad_idx=pad_idx,
                                                   crop_idx=crop_idx,
                                                   n_dims=n_dims,
                                                   labels_segmentation=labels_segmentation,
                                                   strict_masking=strict_masking,
                                                   keep_biggest_component=keep_biggest_component,
                                                   aff=aff,
                                                   im_res=im_res)

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


def prepare_output_files(path_images, path_masks, out_seg, out_posteriors, out_volumes, recompute):

    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert path_masks is not None, 'please specify an input file/folder (--i)'
    assert out_seg is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    path_masks = os.path.abspath(path_masks)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a text file
    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # masks
        if not os.path.isfile(path_masks):
            raise Exception('provided text file containing paths of input images does not exist' % path_masks)
        with open(path_masks, 'r') as f:
            path_masks = [line.replace('\n', '') for line in f.readlines() if line != '\n']

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
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input images
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

        # masks
        if os.path.isfile(path_masks):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_masks)
        path_masks = utils.list_images_in_folder(path_masks)

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
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')

    # path_images is an image
    else:

        # input images
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

        # masks
        assert os.path.isfile(path_masks), 'file does not exist: %s \n' \
                                           'please make sure the path and the extension are correct' % path_masks
        path_masks = [path_masks]

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
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    recompute_list = [recompute | re_seg | re_post | re_vol for (re_seg, re_post, re_vol)
                      in zip(recompute_seg, recompute_post, recompute_volume)]

    return path_images, path_masks, out_seg, out_posteriors, out_volumes, recompute_list, unique_volume_file


def preprocess(path_image, path_mask, n_levels, crop=None, min_pad=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
    mask = utils.load_volume(path_mask, True)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    mask = edit_volumes.align_volume_to_ref(mask, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
        mask = edit_volumes.crop_volume_with_idx(mask, crop_idx, n_dims=n_dims)
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
    mask = edit_volumes.pad_volume(mask, padding_shape=pad_shape)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])
    mask = utils.add_axis(mask, axis=0)  # channel axis will be added later when computing one-hot

    return im, mask, aff, h, im_res, shape, pad_idx, crop_idx


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
                mask_labels):

    assert os.path.isfile(path_model), "The provided model path does not exist."

    # get labels
    n_labels_seg = len(labels_segmentation)

    # one-hot encoding of the input prediction as the network expects soft probabilities
    input_image = KL.Input(input_shape)
    input_labels = KL.Input(input_shape[:-1], dtype='int32')
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=len(mask_labels), axis=-1))(input_labels)
    image = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, labels])
    net = Model(inputs=[input_image, input_labels], outputs=image)

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

    return net


def postprocess(post_patch, mask, shape, pad_idx, crop_idx, n_dims,
                labels_segmentation, strict_masking, keep_biggest_component, aff, im_res):

    # get posteriors
    post_patch = np.squeeze(post_patch)
    mask = np.squeeze(mask)

    # reset posteriors of background to 1 outside mask and to 0 inside
    if strict_masking:
        post_patch[..., 0] = np.ones_like(post_patch[..., 0])
        post_patch[..., 0] = edit_volumes.mask_volume(post_patch[..., 0], mask=mask < 0.1, return_copy=False)
    # keep biggest connected component (use it with smoothing!)
    elif keep_biggest_component:
        tmp_post_patch = post_patch[..., 1:]
        post_patch_mask = np.sum(tmp_post_patch, axis=-1) > 0.25
        post_patch_mask = edit_volumes.get_largest_connected_component(post_patch_mask)
        post_patch_mask = np.stack([post_patch_mask]*tmp_post_patch.shape[-1], axis=-1)
        tmp_post_patch = edit_volumes.mask_volume(tmp_post_patch, mask=post_patch_mask, return_copy=False)
        post_patch[..., 1:] = tmp_post_patch

    # normalise posteriors and get hard segmentation
    if strict_masking | keep_biggest_component:
        post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch.argmax(-1).astype('int32')].astype('int32')

    # paste patches back to matrix of original image size
    seg_patch = edit_volumes.crop_volume_with_idx(seg_patch, pad_idx, n_dims=n_dims, return_copy=False)
    post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=n_dims, return_copy=False)
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
