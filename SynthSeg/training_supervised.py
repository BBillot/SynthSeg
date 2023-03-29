"""

This code is for training is traditional supervised networks with real images and corresponding ground truth labels.
It's relatively simpler than training.py since it here we do not have to generate synthetic scans. However we kept the
parameters for online augmentation.

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
from keras import models
import keras.layers as KL
import numpy.random as npr

# project imports
from SynthSeg import metrics_model as metrics
from SynthSeg.training import train_model
from SynthSeg.labels_to_image_model import get_shapes

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.neuron import models as nrn_models
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im.edit_volumes import get_ras_axes


def training(image_dir,
             labels_dir,
             model_dir,
             segmentation_labels=None,
             n_neutral_labels=None,
             subjects_prob=None,
             batchsize=1,
             target_res=None,
             output_shape=None,
             flipping=True,
             scaling_bounds=.2,
             rotation_bounds=15,
             shearing_bounds=.012,
             translation_bounds=False,
             nonlin_std=4.,
             nonlin_scale=.04,
             randomise_res=True,
             max_res_iso=4.,
             max_res_aniso=8.,
             data_res=None,
             thickness=None,
             bias_field_std=.7,
             bias_scale=.025,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='elu',
             lr=1e-4,
             wl2_epochs=1,
             dice_epochs=50,
             steps_per_epoch=10000,
             checkpoint=None):
    """
    This function trains a UNet to segment MRI images with real scans and corresponding ground truth labels.
    We regroup the parameters in four categories: General, Augmentation, Architecture, Training.

    # IMPORTANT !!!
    # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
    # these values refer to the RAS axes.

    :param image_dir: path of folder with all training images
    :param labels_dir: path of folder with all corresponding label maps
    :param model_dir: path of a directory where the models will be saved during training.

    # ----------------------------------------------- General parameters -----------------------------------------------
    # label maps parameters
    :param n_neutral_labels: (optional) if the label maps contain some right/left specific labels and if flipping is
    applied during training, please provide the number of non-sided labels (including the background).
    This is used to know where the sided labels start in generation_labels. Leave to default (None) if either one of the
    two conditions is not fulfilled.
    :param segmentation_labels: (optional) list of the same length as generation_labels to indicate which values to use
    in the training label maps, i.e. all occurrences of generation_labels[i] in the input label maps will be converted
    to output_labels[i] in the returned label maps. Examples:
    Set output_labels[i] to zero if you wish to erase the value generation_labels[i] from the returned label maps.
    Set output_labels[i]=generation_labels[i] if you wish to keep the value generation_labels[i] in the returned maps.
    Can be a list or a 1d numpy array, or the path to such an array. Default is output_labels = generation_labels.
    :param subjects_prob: (optional) relative order of importance (doesn't have to be probabilistic), with which to pick
    the provided label maps at each minibatch. Can be a sequence, a 1D numpy array, or the path to such an array, and it
    must be as long as path_label_maps. By default, all label maps are chosen with the same importance.

    # output-related parameters
    :param batchsize: (optional) number of images to generate per mini-batch. Default is 1.
    :param target_res: (optional) target resolution at which to teach the network to segment.
    If None, this will be the resolution of the given images/label maps.
    Can be a number (isotropic resolution), or the path to a 1d numpy array.
    :param output_shape: (optional) desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.

    # --------------------------------------------- Augmentation parameters --------------------------------------------
    # spatial deformation parameters
    :param flipping: (optional) whether to introduce right/left random flipping. Default is True.
    :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) the path to a numpy array of shape (2, n_dims), in which case the scaling factor in dimension i is sampled from
    the uniform distribution of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    3) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.2 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for case 1 the
    bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    Default is rotation_bounds = 15.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None).
    :param nonlin_std: (optional) Standard deviation of the normal distribution from which we sample the first
    tensor for synthesising the deformation field. Set to 0 to completely deactivate elastic deformation.
    :param nonlin_scale: (optional) Ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the elastic deformation field.

    # blurring/resampling parameters
    :param randomise_res: (optional) whether to mimic images that would have been 1) acquired at low resolution, and
    2) resampled to high resolution. The low resolution is uniformly resampled at each minibatch from [1mm, 9mm].
    In that process, the images generated by sampling the GMM are: 1) blurred at the sampled LR, 2) downsampled at LR,
    and 3) resampled at target_resolution.
    :param max_res_iso: (optional) If randomise_res is True, this enables to control the upper bound of the uniform
    distribution from which we sample the random resolution U(min_res, max_res_iso), where min_res is the resolution of
    the input label maps. Must be a number, and default is 4. Set to None to deactivate it, but if randomise_res is
    True, at least one of max_res_iso or max_res_aniso must be given.
    :param max_res_aniso: If randomise_res is True, this enables to downsample the input volumes to a random LR in
    only 1 (random) direction. This is done by randomly selecting a direction i in the range [0, n_dims-1], and sampling
    a value in the corresponding uniform distribution U(min_res[i], max_res_aniso[i]), where min_res is the resolution
    of the input label maps. Can be a number, a sequence, or a 1d numpy array. Set to None to deactivate it, but if
    randomise_res is True, at least one of max_res_iso or max_res_aniso must be given.
    :param data_res: (optional) specific acquisition resolution to mimic, as opposed to random resolution sampled when
    randomise_res is True. This triggers a blurring which mimics the acquisition resolution, but downsampling is
    optional (see param downsample). Default for data_res is None, where images are slightly blurred. If the generated
    images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence, a 1d numpy array, or
    the path to a 1d numpy array. In the multi-modal case, it should be given as a numpy array (or a path) of size
    (n_mod, n_dims), where each row is the acquisition resolution of the corresponding channel.
    :param thickness: (optional) if data_res is provided, we can further specify the slice thickness of the low
    resolution images to mimic. Must be provided in the same format as data_res. Default thickness = data_res.

    # bias field parameters
    :param bias_field_std: (optional) If strictly positive, this triggers the corruption of images with a bias field.
    The bias field is obtained by sampling a first small tensor from a normal distribution, resizing it to
    full size, and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the
    std dev of the normal distribution from which we sample the first tensor.
    Set to 0 to completely deactivate bias field corruption.
    :param bias_scale: (optional) If bias_field_std is not False, this designates the ratio between the size of
    the input label maps and the size of the first sampled tensor for synthesising the bias field.

    # ------------------------------------------ UNet architecture parameters ------------------------------------------
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature for the first layer of the UNet. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this number at each new level. Default is 2.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.

    # ----------------------------------------------- Training parameters ----------------------------------------------
    :param lr: (optional) learning rate for the training. Default is 1e-4
    :param wl2_epochs: (optional) number of epochs for which the network (except the soft-max layer) is trained with L2
    norm loss function. Default is 1.
    :param dice_epochs: (optional) number of epochs with the soft Dice loss function. Default is 50.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 10000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param checkpoint: (optional) path of an already saved model to load before starting the training.
    """

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
                                                  nonlin_scale=nonlin_scale,
                                                  randomise_res=randomise_res,
                                                  max_res_iso=max_res_iso,
                                                  max_res_aniso=max_res_aniso,
                                                  data_res=data_res,
                                                  thickness=thickness,
                                                  bias_field_std=bias_field_std,
                                                  bias_scale=bias_scale)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    unet_model = nrn_models.unet(input_model=augmentation_model,
                                 input_shape=unet_input_shape,
                                 nb_labels=n_labels,
                                 nb_levels=n_levels,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_size=conv_size,
                                 nb_features=unet_feat_count,
                                 feat_mult=feat_multiplier,
                                 activation=activation,
                                 batch_norm=-1,
                                 name='unet')

    # input generator
    generator = build_model_inputs(path_images, path_labels, batchsize, subjects_prob)
    input_generator = utils.build_training_generator(generator, batchsize)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics.metrics_model(wl2_model, label_list, 'wl2')
        train_model(wl2_model, input_generator, lr, wl2_epochs, steps_per_epoch, model_dir, 'wl2', checkpoint)
        checkpoint = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    dice_model = metrics.metrics_model(unet_model, label_list, 'dice')
    train_model(dice_model, input_generator, lr, dice_epochs, steps_per_epoch, model_dir, 'dice', checkpoint)


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
                             scaling_bounds=0.2,
                             rotation_bounds=15,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=4.,
                             nonlin_scale=.0625,
                             randomise_res=False,
                             max_res_iso=4.,
                             max_res_aniso=8.,
                             data_res=None,
                             thickness=None,
                             bias_field_std=.7,
                             bias_scale=.025):

    # reformat resolutions and get shapes
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    if data_res is not None:
        data_res = utils.reformat_to_n_channels_array(data_res, n_dims, n_channels)
        thickness = data_res if thickness is None else utils.reformat_to_n_channels_array(thickness, n_dims, n_channels)
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
                                                    nonlin_scale=nonlin_scale,
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
        image = layers.BiasFieldCorruption(bias_field_std, bias_scale, False)(image)

    # intensity augmentation
    image = layers.IntensityAugmentation(6, clip=False, normalise=True, gamma_std=.5, separate_channels=True)(image)

    # if necessary, loop over channels to 1) blur, 2) downsample to simulated LR, and 3) upsample to target
    if data_res is not None:
        channels = list()
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image) if (n_channels > 1) else [image]
        for i, channel in enumerate(split):

            if randomise_res:
                max_res_iso = np.array(utils.reformat_to_list(max_res_iso, length=n_dims, dtype='float'))
                max_res_aniso = np.array(utils.reformat_to_list(max_res_aniso, length=n_dims, dtype='float'))
                max_res = np.maximum(max_res_iso, max_res_aniso)
                resolution, blur_res = layers.SampleResolution(atlas_res, max_res_iso, max_res_aniso)(image)
                sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, resolution, thickness=blur_res)
                channel = layers.DynamicGaussianBlur(0.75 * max_res / np.array(atlas_res), 1.03)([channel, sigma])
                channel = layers.MimicAcquisition(atlas_res, atlas_res, output_shape, False)([channel, resolution])
                channels.append(channel)

            else:
                sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, data_res[i], thickness=thickness[i])
                channel = layers.GaussianBlur(sigma, 1.03)(channel)
                resolution = KL.Lambda(lambda x: tf.convert_to_tensor(data_res[i], dtype='float32'))([])
                channel = layers.MimicAcquisition(atlas_res, data_res[i], output_shape)([channel, resolution])
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


def build_model_inputs(path_inputs,
                       path_outputs,
                       batchsize=1,
                       subjects_prob=None,
                       dtype_input='float32',
                       dtype_output='int32'):

    # get label info
    _, _, _, n_channels, _, _ = utils.get_volume_info(path_inputs[0])

    # make sure subjects_prob sums to 1
    if subjects_prob is not None:
        subjects_prob /= np.sum(subjects_prob)

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.choice(np.arange(len(path_outputs)), size=batchsize, p=subjects_prob)

        # initialise input lists
        list_batch_inputs = list()
        list_batch_outputs = list()

        for idx in indices:

            # get a batch input
            batch_input = utils.load_volume(path_inputs[idx], aff_ref=np.eye(4), dtype=dtype_input)
            if n_channels > 1:
                list_batch_inputs.append(utils.add_axis(batch_input, axis=0))
            else:
                list_batch_inputs.append(utils.add_axis(batch_input, axis=[0, -1]))

            # get a batch output
            batch_output = utils.load_volume(path_outputs[idx], aff_ref=np.eye(4), dtype=dtype_output)
            list_batch_outputs.append(utils.add_axis(batch_output, axis=[0, -1]))

        # build list of training pairs
        list_training_pairs = [list_batch_inputs, list_batch_outputs]
        if batchsize > 1:  # concatenate individual input types if batchsize > 1
            list_training_pairs = [np.concatenate(item, 0) for item in list_training_pairs]
        else:
            list_training_pairs = [item[0] for item in list_training_pairs]

        yield list_training_pairs
