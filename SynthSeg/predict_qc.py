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

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models


def predict(path_predictions,
            path_qc_results,
            path_model,
            labels_list,
            labels_to_convert=None,
            convert_gt=False,
            shape=224,
            n_levels=5,
            nb_conv_per_level=3,
            conv_size=5,
            unet_feat_count=24,
            feat_multiplier=2,
            activation='relu',
            path_gts=None,
            verbose=True):

    # prepare input/output filepaths
    path_predictions, path_gts, path_qc_results, path_gt_results, path_diff = \
        prepare_output_files(path_predictions, path_gts, path_qc_results)

    # get label list
    labels_list, _ = utils.get_list_labels(label_list=labels_list)
    labels_list_unique, _ = np.unique(labels_list, return_index=True)
    if labels_to_convert is not None:
        labels_to_convert, _ = utils.get_list_labels(label_list=labels_to_convert)

    # prepare qc results
    pred_qc_results = np.zeros((len(labels_list_unique) + 1, len(path_predictions)))
    gt_qc_results = np.zeros((len(labels_list_unique), len(path_predictions))) if path_gt_results is not None else None

    # build network
    model_input_shape = [None, None, None, 1]
    net = build_qc_model(path_model=path_model,
                         input_shape=model_input_shape,
                         label_list=labels_list_unique,
                         n_levels=n_levels,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_size=conv_size,
                         unet_feat_count=unet_feat_count,
                         feat_multiplier=feat_multiplier,
                         activation=activation)

    # perform segmentation
    loop_info = utils.LoopInfo(len(path_predictions), 10, 'predicting', True)
    for idx, (path_prediction, path_gt) in enumerate(zip(path_predictions, path_gts)):

        # compute segmentation only if needed
        if verbose:
            loop_info.update(idx)

        # preprocessing
        prediction, gt_scores = preprocess(path_prediction, path_gt, shape, labels_list, labels_to_convert, convert_gt)

        # get predicted scores
        pred_qc_results[-1, idx] = np.sum(prediction > 0)
        pred_qc_results[:-1, idx] = np.clip(np.squeeze(net.predict(prediction)), 0, 1)
        np.save(path_qc_results, pred_qc_results)

        # save GT scores if necessary
        if gt_scores is not None:
            gt_qc_results[:, idx] = gt_scores
            np.save(path_gt_results, gt_qc_results)

    if path_diff is not None:
        diff = pred_qc_results[:-1, :] - gt_qc_results
        np.save(path_diff, diff)


def prepare_output_files(path_predictions, path_gts, path_qc_results):

    # check inputs
    assert path_predictions is not None, 'please specify an input file/folder (--i)'
    assert path_qc_results is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_predictions = os.path.abspath(path_predictions)
    path_qc_results = os.path.abspath(path_qc_results)

    # list input predictions
    path_predictions = utils.list_images_in_folder(path_predictions)

    # build path output with qc results
    if path_qc_results[-4:] != '.npy':
        print('Path for QC outputs provided without npy extension. Adding npy extension.')
        path_qc_results += '.npy'
    utils.mkdir(os.path.dirname(path_qc_results))

    if path_gts is not None:
        path_gts = utils.list_images_in_folder(path_gts)
        assert len(path_gts) == len(path_predictions), 'not the same number of predictions and GTs'
        path_gt_results = path_qc_results.replace('.npy', '_gt.npy')
        path_diff = path_qc_results.replace('.npy', '_diff.npy')
    else:
        path_gts = [None] * len(path_predictions)
        path_gt_results = path_diff = None

    return path_predictions, path_gts, path_qc_results, path_gt_results, path_diff


def preprocess(path_prediction, path_gt=None, shape=224, labels_list=None, labels_to_convert=None, convert_gt=False):

    # read image and corresponding info
    pred, _, aff_pred, n_dims, _, _, _ = utils.get_volume_info(path_prediction, True)
    gt = utils.load_volume(path_gt, aff_ref=np.eye(4)) if path_gt is not None else None

    # align
    pred = edit_volumes.align_volume_to_ref(pred, aff_pred, aff_ref=np.eye(4), n_dims=n_dims)

    # pad/crop to 224, such that segmentations are in the middle of the patch
    if gt is not None:
        pred, gt = make_shape(pred, gt, shape, n_dims)
    else:
        pred, _ = edit_volumes.crop_volume_around_region(pred, cropping_shape=shape)

    # convert labels if necessary
    if labels_to_convert is not None:
        lut = utils.get_mapping_lut(labels_to_convert, labels_list)
        pred = lut[pred.astype('int32')]
        if convert_gt & (gt is not None):
            gt = lut[gt.astype('int32')]

    # compute GT dice scores
    gt_scores = evaluate.fast_dice(pred, gt, np.unique(labels_list)) if gt is not None else None

    # add batch and channel axes
    pred = utils.add_axis(pred, axis=0)  # channel axis will be added later when computing one-hot

    return pred, gt_scores


def make_shape(pred, gt, shape, n_dims):

    mask = ((pred > 0) & (pred != 24)) | (gt > 0)
    vol_shape = np.array(pred.shape[:n_dims])

    if np.any(mask):

        # find cropping indices
        indices = np.nonzero(mask)
        min_idx = np.maximum(np.array([np.min(idx) for idx in indices]), 0)
        max_idx = np.minimum(np.array([np.max(idx) for idx in indices]) + 1, vol_shape)

        # expand/retract (depending on the desired shape) the cropping region around the centre
        intermediate_vol_shape = max_idx - min_idx
        cropping_shape = np.array(utils.reformat_to_list(shape, length=n_dims))
        min_idx = min_idx - np.int32(np.ceil((cropping_shape - intermediate_vol_shape) / 2))
        max_idx = max_idx + np.int32(np.floor((cropping_shape - intermediate_vol_shape) / 2))

        # crop volume
        cropping = np.concatenate([np.maximum(min_idx, 0), np.minimum(max_idx, vol_shape)])
        pred = edit_volumes.crop_volume_with_idx(pred, cropping, n_dims=n_dims)
        gt = edit_volumes.crop_volume_with_idx(gt, cropping, n_dims=n_dims)

        # check if we need to pad the output to the desired shape
        min_padding = np.abs(np.minimum(min_idx, 0))
        max_padding = np.maximum(max_idx - vol_shape, 0)
        if np.any(min_padding > 0) | np.any(max_padding > 0):
            pad_margins = tuple([(min_padding[i], max_padding[i]) for i in range(n_dims)])
            pred = np.pad(pred, pad_margins, mode='constant', constant_values=0)
            gt = np.pad(gt, pad_margins, mode='constant', constant_values=0)

    return pred, gt


def build_qc_model(path_model,
                   input_shape,
                   label_list,
                   n_levels,
                   nb_conv_per_level,
                   conv_size,
                   unet_feat_count,
                   feat_multiplier,
                   activation):

    assert os.path.isfile(path_model), "The provided model path does not exist."
    label_list_unique = np.unique(label_list)
    n_labels = len(label_list_unique)

    # one-hot encoding of the input prediction as the network expects soft probabilities
    input_labels = KL.Input(input_shape[:-1])
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(input_labels)
    net = Model(inputs=input_labels, outputs=labels)

    # build model
    model = nrn_models.conv_enc(input_model=net,
                                input_shape=input_shape,
                                nb_levels=n_levels,
                                nb_conv_per_level=nb_conv_per_level,
                                conv_size=conv_size,
                                nb_features=unet_feat_count,
                                feat_mult=feat_multiplier,
                                activation=activation,
                                batch_norm=-1,
                                use_residuals=True,
                                name='qc')
    last = model.outputs[0]

    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    last = KL.MaxPool3D(pool_size=(2, 2, 2), name='qc_maxpool_%s' % (n_levels - 1), padding='same')(last)
    last = KL.Conv3D(n_labels, kernel_size=5, **conv_kwargs, name='qc_final_conv_0')(last)
    last = KL.Conv3D(n_labels, kernel_size=5, **conv_kwargs, name='qc_final_conv_1')(last)
    last = KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='qc_final_pred')(last)

    net = Model(inputs=net.inputs, outputs=last)
    net.load_weights(path_model, by_name=True)

    return net
