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
import copy
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL

# third-party imports
from ext.lab2im import layers as layers
from ext.lab2im import utils, edit_volumes
from ext.neuron import models as nrn_models
from ext.lab2im import edit_tensors as l2i_et


def sample_segmentation_pairs(image_dir,
                              labels_dir,
                              results_dir,
                              n_examples,
                              path_model,
                              segmentation_labels,
                              n_neutral_labels=None,
                              batchsize=1,
                              flipping=True,
                              scaling_bounds=.15,
                              rotation_bounds=15,
                              shearing_bounds=.012,
                              translation_bounds=False,
                              nonlin_std=3.,
                              nonlin_scale=.04,
                              min_res=1.,
                              max_res_iso=4.,
                              max_res_aniso=8.,
                              noise_std_lr=3.,
                              blur_range=1.03,
                              bias_field_std=.5,
                              bias_scale=.025,
                              noise_std=10,
                              gamma_std=.5):

    """
    This function enables us to obtain segmentations from a segmentation network along with the corresponding ground
    truths. The segmentations are obtained by taking real images and aggressively augmenting them with spatial and
    intensity transforms, such that the network commits some mistakes. The sample pairs can then be used to train a
    denoising network by using the noisy segmentations as inputs and the ground truth as targets.

    # IMPORTANT !!!
    # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
    # these values refer to the RAS axes.

    :param image_dir: path of folder with all images
    :param labels_dir: path of folder with all ground truth segmentations
    :param results_dir: path of a directory where the results will be saved
    :param n_examples: number of pairs to sample
    :param path_model: path of the trained model to obtain segmentations from

    # --------------------------------------------- Degradation parameters ---------------------------------------------
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

    # output-related parameters
    :param batchsize: (optional) number of images to generate per mini-batch. Default is 1.

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
    :param min_res: (optional) resolution of the input pairs of images/segmentations
    :param max_res_iso: (optional) This enables to control the upper bound of the uniform distribution from which we
    sample the random resolution U(min_res, max_res_iso), where min_res is the resolution of the input label maps.
    Must be a number, and default is 4.
    :param max_res_aniso: This enables to downsample the input volumes to a random LR in only 1 (random) direction. This
    is done by randomly selecting a direction i in the range [0, n_dims-1], and sampling a value in the corresponding
    uniform distribution U(min_res[i], max_res_aniso[i]), where min_res is the resolution of the input label maps. Can
    be a number, a sequence, or a 1d numpy array.
    :param blur_range: (optional) coef to randomise the blurring kernel

    # bias field parameters
    :param bias_field_std: (optional) If strictly positive, this triggers the corruption of images with a bias field.
    The bias field is obtained by sampling a first small tensor from a normal distribution, resizing it to
    full size, and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the
    std dev of the normal distribution from which we sample the first tensor.
    Set to 0 to completely deactivate bias field corruption.
    :param bias_scale: (optional) If bias_field_std is not False, this designates the ratio between the size of
    the input label maps and the size of the first sampled tensor for synthesising the bias field.

    # noise parameters
    :param noise_std: (optional) standard deviation of the white Gaussian noise added at high resolution to the image.
    :param noise_std_lr: (optional) sta dev of the white Gaussian noise added at low resolution to the image.
    :param gamma_std: (optional) standard deviation of the gaussian transform to apply at high resolution to the image.
    """

    # prepare data files
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), "There should be as many images as label maps."

    # prepare results subfolders
    gt_result_dir = os.path.join(results_dir, 'labels_gt')
    pred_result_dir = os.path.join(results_dir, 'labels_seg')
    utils.mkdir(gt_result_dir)
    utils.mkdir(pred_result_dir)

    # get label lists
    segmentation_labels, _ = utils.get_list_labels(label_list=segmentation_labels, labels_dir=labels_dir)
    n_labels = np.size(np.unique(segmentation_labels))

    # create augmentation model
    im_shape, _, n_dims, n_channels, _, atlas_res = utils.get_volume_info(path_images[0], aff_ref=np.eye(4))
    augmentation_model = build_augmentation_model(im_shape,
                                                  n_channels,
                                                  segmentation_labels,
                                                  n_neutral_labels,
                                                  n_dims,
                                                  atlas_res,
                                                  flipping=flipping,
                                                  aff=np.eye(4),
                                                  scaling_bounds=scaling_bounds,
                                                  rotation_bounds=rotation_bounds,
                                                  shearing_bounds=shearing_bounds,
                                                  translation_bounds=translation_bounds,
                                                  nonlin_std=nonlin_std,
                                                  nonlin_shape_factor=nonlin_scale,
                                                  min_res=min_res,
                                                  max_res_iso=max_res_iso,
                                                  max_res_aniso=max_res_aniso,
                                                  noise_std_lr=noise_std_lr,
                                                  blur_range=blur_range,
                                                  bias_field_std=bias_field_std,
                                                  bias_shape_factor=bias_scale,
                                                  noise_std=noise_std,
                                                  gamma_std=gamma_std)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    unet_model = nrn_models.unet(nb_features=24,
                                 input_shape=unet_input_shape,
                                 nb_levels=5,
                                 conv_size=3,
                                 nb_labels=n_labels,
                                 feat_mult=2,
                                 nb_conv_per_level=2,
                                 batch_norm=-1,
                                 activation='elu',
                                 input_model=augmentation_model)

    # get generative model
    train_example_gen = build_model_inputs(path_images, path_labels, batchsize)
    segmentation_labels = np.unique(segmentation_labels)

    # redefine model to output deformed image, deformed GT labels, and predicted labels
    list_output_tensors = [unet_model.get_layer('labels_out').output, unet_model.output]
    generation_model = models.Model(inputs=unet_model.inputs, outputs=list_output_tensors)
    generation_model.load_weights(path_model, by_name=True)

    # generate !
    n = len(str(n_examples))
    i = 1
    while i <= n_examples:

        # predict new segmentation
        outputs = generation_model.predict(next(train_example_gen))

        # save results
        for (output, name, res_dir) in zip(outputs,
                                           ['labels_gt', 'labels_pred_argmax_convert'],
                                           [gt_result_dir, pred_result_dir]):
            for b in range(batchsize):
                tmp_name = copy.deepcopy(name)
                tmp_output = np.squeeze(output[b, ...])
                if '_argmax' in tmp_name:
                    tmp_output = tmp_output.argmax(-1)
                    tmp_name = tmp_name.replace('_argmax', '')
                if '_convert' in tmp_name:
                    tmp_output = segmentation_labels[tmp_output]
                    tmp_name = tmp_name.replace('_convert', '')
                path = os.path.join(res_dir, tmp_name + '_%.{}d'.format(n) % i + '.nii.gz')
                if batchsize > 1:
                    path = path.replace('.nii.gz', '_%s.nii.gz' % (b + 1))
                utils.save_volume(tmp_output, np.eye(4), None, path)

        i += 1


def build_augmentation_model(im_shape,
                             n_channels,
                             segmentation_labels,
                             n_neutral_labels,
                             n_dims,
                             atlas_res,
                             flipping=True,
                             aff=None,
                             scaling_bounds=0.15,
                             rotation_bounds=15,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=3.,
                             nonlin_shape_factor=.0625,
                             min_res=1.,
                             max_res_iso=4.,
                             max_res_aniso=8.,
                             noise_std_lr=3.,
                             blur_range=1.03,
                             bias_field_std=.5,
                             bias_shape_factor=.025,
                             noise_std=10,
                             gamma_std=.5):

    # define model inputs
    image_input = KL.Input(shape=im_shape+[n_channels], name='image_input')
    labels_input = KL.Input(shape=im_shape + [1], name='labels_input', dtype='int32')

    # deform labels
    labels, image = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                    rotation_bounds=rotation_bounds,
                                                    shearing_bounds=shearing_bounds,
                                                    translation_bounds=translation_bounds,
                                                    nonlin_std=nonlin_std,
                                                    nonlin_scale=nonlin_shape_factor,
                                                    inter_method=['nearest', 'linear'])([labels_input, image_input])

    # flipping
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        labels, image = layers.RandomFlip(edit_volumes.get_ras_axes(aff, n_dims)[0], [True, False],
                                          segmentation_labels, n_neutral_labels)([labels, image])

    # apply bias field
    if bias_field_std > 0:
        image = layers.BiasFieldCorruption(bias_field_std, bias_shape_factor, False)(image)

    # intensity augmentation
    image = layers.IntensityAugmentation(noise_std, gamma_std=gamma_std, contrast_inversion=True)(image)

    # loop over channels
    channels = list()
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image) if (n_channels > 1) else [image]
    for i, channel in enumerate(split):

        # reformat resolution range parameters
        min_res = np.array(utils.reformat_to_list(min_res, length=n_dims, dtype='float'))
        max_res_iso = np.array(utils.reformat_to_list(max_res_iso, length=n_dims, dtype='float'))
        max_res_aniso = np.array(utils.reformat_to_list(max_res_aniso, length=n_dims, dtype='float'))
        max_res = np.maximum(max_res_iso, max_res_aniso)

        # sample resolution and thickness (blurring res)
        resolution, blur_res = layers.SampleResolution(min_res, max_res_iso, max_res_aniso)(channel)
        sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, resolution, thickness=blur_res)

        # blur and downsample/resample
        channel = layers.DynamicGaussianBlur(0.75 * max_res / np.array(atlas_res), blur_range)([channel, sigma])
        channel = layers.MimicAcquisition(atlas_res, min_res, im_shape, False, noise_std_lr)([channel, resolution])
        channels.append(channel)

    # concatenate all channels back
    image = KL.Lambda(lambda x: tf.concat(x, -1))(channels) if len(channels) > 1 else channels[0]

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
        indices = np.random.randint(len(path_label_maps), size=batchsize)

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
            labels = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(labels, axis=[0, -1]))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate individual input types if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
