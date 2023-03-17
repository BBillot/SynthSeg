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
from SynthSeg.predict import write_csv, postprocess

# third-party imports
from ext.lab2im import edit_volumes
from ext.lab2im import utils, layers
from ext.neuron import models as nrn_models


def predict(path_predictions,
            path_corrections,
            path_model,
            input_segmentation_labels,
            target_segmentation_labels=None,
            names_segmentation=None,
            path_posteriors=None,
            path_volumes=None,
            min_pad=None,
            cropping=None,
            topology_classes=None,
            sigma_smoothing=0.5,
            keep_biggest_component=True,
            n_levels=5,
            nb_conv_per_level=2,
            conv_size=5,
            unet_feat_count=16,
            feat_multiplier=2,
            activation='elu',
            skip_n_concatenations=2,
            gt_folder=None,
            evaluation_labels=None,
            list_incorrect_labels=None,
            list_correct_labels=None,
            compute_distances=False,
            recompute=True,
            verbose=True):

    # prepare input/output filepaths
    path_predictions, path_corrections, path_posteriors, path_volumes, compute, unique_vol_file = \
        prepare_output_files(path_predictions, path_corrections, path_posteriors, path_volumes, recompute)

    # get label list
    input_segmentation_labels = utils.get_list_labels(label_list=input_segmentation_labels)[0]
    input_segmentation_labels, unique_idx = np.unique(input_segmentation_labels, return_index=True)
    if target_segmentation_labels is not None:
        target_segmentation_labels = utils.get_list_labels(label_list=target_segmentation_labels)[0]
        target_segmentation_labels, unique_idx = np.unique(target_segmentation_labels, return_index=True)
    else:
        target_segmentation_labels = input_segmentation_labels

    # prepare other labels list
    if names_segmentation is not None:
        names_segmentation = utils.load_array_if_path(names_segmentation)[unique_idx]
    if topology_classes is not None:
        topology_classes = utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]

    # prepare volumes if necessary
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, target_segmentation_labels, names_segmentation)

    # build network
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_predictions[0])
    model_input_shape = [None] * n_dims + [n_channels]
    net = build_model(path_model=path_model,
                      input_shape=model_input_shape,
                      input_label_list=input_segmentation_labels,
                      target_label_list=target_segmentation_labels,
                      n_levels=n_levels,
                      nb_conv_per_level=nb_conv_per_level,
                      conv_size=conv_size,
                      unet_feat_count=unet_feat_count,
                      feat_multiplier=feat_multiplier,
                      activation=activation,
                      skip_n_concatenations=skip_n_concatenations,
                      sigma_smoothing=sigma_smoothing)

    # set cropping/padding
    if (cropping is not None) & (min_pad is not None):
        cropping = utils.reformat_to_list(cropping, length=n_dims, dtype='int')
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = np.minimum(cropping, min_pad)

    # perform segmentation
    if len(path_predictions) <= 10:
        loop_info = utils.LoopInfo(len(path_predictions), 1, 'predicting', True)
    else:
        loop_info = utils.LoopInfo(len(path_predictions), 10, 'predicting', True)
    for i in range(len(path_predictions)):
        if verbose:
            loop_info.update(i)

        # compute segmentation only if needed
        if compute[i]:

            # preprocessing
            prediction, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_prediction=path_predictions[i],
                                                                              n_levels=n_levels,
                                                                              crop=cropping,
                                                                              min_pad=min_pad)

            # prediction
            post_patch = net.predict(prediction)

            # postprocessing
            seg, posteriors, volumes = postprocess(post_patch=post_patch,
                                                   shape=shape,
                                                   pad_idx=pad_idx,
                                                   crop_idx=crop_idx,
                                                   n_dims=n_dims,
                                                   labels_segmentation=target_segmentation_labels,
                                                   keep_biggest_component=keep_biggest_component,
                                                   aff=aff,
                                                   im_res=im_res,
                                                   topology_classes=topology_classes)

            # write results to disk
            utils.save_volume(seg, aff, h, path_corrections[i], dtype='int32')
            if path_posteriors[i] is not None:
                if n_channels > 1:
                    posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')

            # compute volumes
            if path_volumes[i] is not None:
                row = [os.path.basename(path_predictions[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, target_segmentation_labels, names_segmentation)

    # evaluate
    if gt_folder is not None:

        # find path where segmentations are saved evaluation folder, and get labels on which to evaluate
        eval_folder = os.path.dirname(path_corrections[0])
        if evaluation_labels is None:
            evaluation_labels = target_segmentation_labels

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


def prepare_output_files(path_predictions, out_corrections, out_posteriors, out_volumes, recompute):

    # check inputs
    assert path_predictions is not None, 'please specify an input file/folder (--i)'
    assert out_corrections is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_predictions = os.path.abspath(path_predictions)
    basename = os.path.basename(path_predictions)
    out_corrections = os.path.abspath(out_corrections)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a text file
    if basename[-4:] == '.txt':

        # input predictions
        if not os.path.isfile(path_predictions):
            raise Exception('provided text file containing paths of input prediction does not exist' % path_predictions)
        with open(path_predictions, 'r') as f:
            path_predictions = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # define helper to deal with outputs
        def text_helper(path, name):
            if path is not None:
                assert path[-4:] == '.txt', 'if path_predictions given as text file, so must be %s' % name
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_predictions)
                recompute_files = [False] * len(path_predictions)
            unique_file = False
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_corrections, recompute_cor, _ = text_helper(out_corrections, 'path_corrections')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')

    # path_predictions is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input predictions
        if os.path.isfile(path_predictions):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_predictions)
        path_predictions = utils.list_images_in_folder(path_predictions)

        # define helper to deal with outputs
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_predictions is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    path = [path] * len(path_predictions)
                    recompute_files = [True] * len(path_predictions)
                    unique_file = True
                else:
                    if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (path[-4:] == '.npz'):
                        raise Exception('Output FOLDER had a FILE extension' % path)
                    path = [os.path.join(path, os.path.basename(p)) for p in path_predictions]
                    path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                    path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                    path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                utils.mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_predictions)
                recompute_files = [False] * len(path_predictions)
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_corrections, recompute_cor, _ = helper_dir(out_corrections, 'path_corrections', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_dir(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')

    # path_predictions is an image
    else:

        # input prediction
        assert os.path.isfile(path_predictions), 'file does not exist: %s \nplease make sure the path and ' \
                                                 'the extension are correct' % path_predictions
        path_predictions = [path_predictions]

        # define helper to deal with outputs
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_predictions is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                        file_name = os.path.basename(path_predictions[0]).replace('.nii', '_%s.nii' % suffix)
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
        out_corrections, recompute_cor, _ = helper_im(out_corrections, 'path_corrections', '', 'synthseg')
        out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    recompute_list = [recompute | re_cor | re_post | re_vol for (re_cor, re_post, re_vol)
                      in zip(recompute_cor, recompute_post, recompute_volume)]

    return path_predictions, out_corrections, out_posteriors, out_volumes, recompute_list, unique_volume_file


def preprocess(path_prediction, n_levels, crop=None, min_pad=None):

    # read image and corresponding info
    pred, _, aff_pred, n_dims, _, h_pred, res_pred = utils.get_volume_info(path_prediction, True)

    # align image
    pred = edit_volumes.align_volume_to_ref(pred, aff_pred, aff_ref=np.eye(4), n_dims=n_dims)
    shape = list(pred.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        input_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        pred, crop_idx = edit_volumes.crop_volume(pred, cropping_shape=input_shape, return_crop_idx=True)
    else:
        crop_idx = None
        input_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in shape]

    # pad image
    if min_pad is not None:  # in SynthSeg predict use crop flag and then if used do min_pad=crop else min_pad = 192
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        input_shape = np.maximum(input_shape, min_pad)
    pred, pad_idx = edit_volumes.pad_volume(pred, padding_shape=input_shape, return_pad_idx=True)

    # add batch and channel axes
    pred = utils.add_axis(pred, axis=0)  # channel axis will be added later when computing one-hot

    return pred, aff_pred, h_pred, res_pred, shape, pad_idx, crop_idx


def build_model(path_model,
                input_shape,
                input_label_list,
                target_label_list,
                n_levels,
                nb_conv_per_level,
                conv_size,
                unet_feat_count,
                feat_multiplier,
                activation,
                skip_n_concatenations,
                sigma_smoothing):

    assert os.path.isfile(path_model), "The provided model path does not exist."

    # get labels
    n_labels_seg = len(target_label_list)

    # one-hot encoding of the input prediction as the network expects soft probabilities
    input_labels = KL.Input(input_shape[:-1])
    labels = layers.ConvertLabels(input_label_list)(input_labels)
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=len(input_label_list), axis=-1))(labels)
    net = Model(inputs=input_labels, outputs=labels)

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
                          batch_norm=-1,
                          skip_n_concatenations=skip_n_concatenations,
                          name='l2l')
    net.load_weights(path_model, by_name=True)

    # smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

    return net
