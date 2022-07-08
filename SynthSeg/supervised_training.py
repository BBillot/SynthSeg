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
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL
import numpy.random as npr

# project imports
from SynthSeg.training import train_model
from SynthSeg import metrics_model as metrics
from SynthSeg.labels_to_image_model import get_shapes

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.neuron import models as nrn_models
from ext.neuron import layers as nrn_layers
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im.edit_volumes import get_ras_axes


def supervised_training(image_dir,
                        labels_dir,
                        model_dir,
                        segmentation_labels=None,
                        n_neutral_labels=None,
                        batchsize=1,
                        target_res=None,
                        output_shape=None,
                        flipping=True,
                        scaling_bounds=.15,
                        rotation_bounds=15,
                        shearing_bounds=.012,
                        translation_bounds=False,
                        nonlin_std=3.,
                        nonlin_shape_factor=.04,
                        data_res=None,
                        thickness=None,
                        downsample=False,
                        blur_range=1.03,
                        bias_field_std=.5,
                        bias_shape_factor=.025,
                        n_levels=5,
                        nb_conv_per_level=2,
                        conv_size=3,
                        unet_feat_count=24,
                        feat_multiplier=2,
                        activation='elu',
                        lr=1e-4,
                        lr_decay=0,
                        wl2_epochs=5,
                        dice_epochs=100,
                        steps_per_epoch=1000,
                        checkpoint=None):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # prepare data files
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), "There should be as many images as label maps."

    # get label lists
    label_list, _ = utils.get_list_labels(label_list=segmentation_labels, labels_dir=labels_dir)
    n_labels = np.size(label_list)

    # create augmentation model
    im_shape, _, _, n_channels, _, atlas_res = utils.get_volume_info(path_images[0], aff_ref=np.eye(4))
    augmentation_model = build_augmentation_model(im_shape,
                                                  n_channels,
                                                  label_list,
                                                  n_neutral_labels,
                                                  atlas_res,
                                                  target_res,
                                                  output_shape=output_shape,
                                                  output_div_by_n=2 ** n_levels,
                                                  flipping=flipping,
                                                  aff=np.eye(4),
                                                  scaling_bounds=scaling_bounds,
                                                  rotation_bounds=rotation_bounds,
                                                  shearing_bounds=shearing_bounds,
                                                  translation_bounds=translation_bounds,
                                                  nonlin_std=nonlin_std,
                                                  nonlin_shape_factor=nonlin_shape_factor,
                                                  data_res=data_res,
                                                  thickness=thickness,
                                                  downsample=downsample,
                                                  blur_range=blur_range,
                                                  bias_field_std=bias_field_std,
                                                  bias_shape_factor=bias_shape_factor)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=augmentation_model)

    # input generator
    input_generator = utils.build_training_generator(build_model_inputs(path_images, path_labels, batchsize), batchsize)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics.metrics_model(wl2_model, label_list, 'wl2')
        train_model(wl2_model, input_generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, 'wl2', checkpoint)
        checkpoint = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    dice_model = metrics.metrics_model(unet_model, label_list, 'dice')
    train_model(dice_model, input_generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, 'dice', checkpoint)


def build_augmentation_model(im_shape,
                             n_channels,
                             segmentation_labels,
                             n_neutral_labels,
                             atlas_res,
                             target_res,
                             output_shape=None,
                             output_div_by_n=None,
                             flipping=True,
                             aff=None,
                             scaling_bounds=0.15,
                             rotation_bounds=15,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=3.,
                             nonlin_shape_factor=.0625,
                             data_res=None,
                             thickness=None,
                             downsample=False,
                             blur_range=1.03,
                             bias_field_std=.5,
                             bias_shape_factor=.025):

    # reformat resolutions and get shapes
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    if data_res is not None:
        data_res = utils.reformat_to_n_channels_array(data_res, n_dims, n_channels)
        thickness = data_res if thickness is None else utils.reformat_to_n_channels_array(thickness, n_dims, n_channels)
        downsample = utils.reformat_to_list(downsample, n_channels) if downsample else np.min(thickness-data_res, 1) < 0
        target_res = atlas_res if (target_res is None) else utils.reformat_to_n_channels_array(target_res, n_dims)[0]
    else:
        target_res = atlas_res

    # get shapes
    crop_shape, output_shape = get_shapes(im_shape, output_shape, atlas_res, target_res, output_div_by_n)

    # define model inputs
    image_input = KL.Input(shape=im_shape+[n_channels], name='image_input')
    labels_input = KL.Input(shape=im_shape + [1], name='labels_input', dtype='int32')

    # deform labels
    labels, image = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                    rotation_bounds=rotation_bounds,
                                                    shearing_bounds=shearing_bounds,
                                                    translation_bounds=translation_bounds,
                                                    nonlin_std=nonlin_std,
                                                    nonlin_shape_factor=nonlin_shape_factor,
                                                    inter_method=['nearest', 'linear'])([labels_input, image_input])

    # cropping
    if crop_shape != im_shape:
        labels, image = layers.RandomCrop(crop_shape)([labels, image])

    # flipping
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        labels, image = layers.RandomFlip(get_ras_axes(aff, n_dims)[0], [True, False],
                                          segmentation_labels, n_neutral_labels)([labels, image])

    # apply bias field
    if bias_field_std > 0:
        image = layers.BiasFieldCorruption(bias_field_std, bias_shape_factor, False)(image)

    # intensity augmentation
    image = layers.IntensityAugmentation(6, clip=False, normalise=True, gamma_std=.4, separate_channels=True)(image)

    # if necessary, loop over channels to 1) blur, 2) downsample to simulated LR, and 3) upsample to target
    if data_res is not None:
        channels = list()
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image) if (n_channels > 1) else [image]
        for i, channel in enumerate(split):

            # blur
            sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, data_res[i], thickness=thickness[i])
            channel = layers.GaussianBlur(sigma, blur_range)(channel)

            # resample
            if downsample[i]:
                resolution = KL.Lambda(lambda x: tf.convert_to_tensor(data_res[i], dtype='float32'))([])
                channel = layers.MimicAcquisition(atlas_res, data_res[i], output_shape)([channel, resolution])
            elif output_shape != crop_shape:
                channel = nrn_layers.Resize(size=output_shape)(channel)
            channels.append(channel)

        # concatenate all channels back
        image = KL.Lambda(lambda x: tf.concat(x, -1))(channels) if len(channels) > 1 else channels[0]

        # resample labels at target resolution
        if crop_shape != output_shape:
            labels = l2i_et.resample_tensor(labels, output_shape, interp_method='nearest')

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'), name='labels_out')(labels)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])
    brain_model = models.Model(inputs=[image_input, labels_input], outputs=[image, labels])

    return brain_model


def build_model_inputs(path_images, path_label_maps, batchsize=1):

    # get label info
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_images[0])

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_label_maps), size=batchsize)

        # initialise input lists
        list_images = list()
        list_label_maps = list()

        for idx in indices:

            # add image
            image = utils.load_volume(path_images[idx], aff_ref=np.eye(4))
            if n_channels > 1:
                list_images.append(utils.add_axis(image, axis=0))
            else:
                list_images.append(utils.add_axis(image, axis=[0, -1]))

            # add labels
            labels = utils.load_volume(path_label_maps[idx], dtype='int32', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(labels, axis=[0, -1]))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate individual input types if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
