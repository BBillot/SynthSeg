# python imports
import os
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import numpy.random as npr
from keras.models import Model

# project imports
from SynthSeg import metrics_model
from SynthSeg.training import train_model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models
from ext.neuron import layers as nrn_layers
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im import spatial_augmentation as l2i_sa
from ext.lab2im import intensity_augmentation as l2i_ia


def supervised_training(image_dir,
                        labels_dir,
                        model_dir,
                        path_segmentation_labels=None,
                        batchsize=1,
                        output_shape=None,
                        flipping=True,
                        apply_linear_trans=True,
                        scaling_bounds=None,
                        rotation_bounds=None,
                        shearing_bounds=None,
                        apply_nonlin_trans=True,
                        nonlin_std=3.,
                        nonlin_shape_factor=.04,
                        crop_channel_2=None,
                        apply_bias_field=True,
                        bias_field_std=.3,
                        bias_shape_factor=.025,
                        n_levels=5,
                        nb_conv_per_level=2,
                        conv_size=3,
                        unet_feat_count=24,
                        feat_multiplier=2,
                        dropout=0,
                        activation='elu',
                        lr=1e-4,
                        lr_decay=0,
                        wl2_epochs=5,
                        dice_epochs=100,
                        steps_per_epoch=1000,
                        load_model_file=None,
                        initial_epoch_wl2=0,
                        initial_epoch_dice=0):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # prepare data files
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), "There should be as many images as label maps."

    # get label lists
    label_list, n_neutral_labels = utils.get_list_labels(label_list=path_segmentation_labels, labels_dir=labels_dir,
                                                         FS_sort=True)
    n_labels = np.size(label_list)

    # prepare model folder
    utils.mkdir(model_dir)

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # create augmentation model and input generator
    im_shape, _, _, n_channels, _, _ = utils.get_volume_info(path_images[0], aff_ref=np.eye(4))
    augmentation_model = labels_to_image_model(im_shape,
                                               batchsize,
                                               n_channels,
                                               label_list,
                                               n_neutral_labels,
                                               output_shape=output_shape,
                                               output_div_by_n=2 ** n_levels,
                                               flipping=flipping,
                                               aff=np.eye(4),
                                               apply_linear_trans=apply_linear_trans,
                                               apply_nonlin_trans=apply_nonlin_trans,
                                               nonlin_std=nonlin_std,
                                               nonlin_shape_factor=nonlin_shape_factor,
                                               crop_channel2=crop_channel_2,
                                               apply_bias_field=apply_bias_field,
                                               bias_field_std=bias_field_std,
                                               bias_shape_factor=bias_shape_factor)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    model_input_generator = build_model_inputs(path_images,
                                               path_labels,
                                               batchsize=batchsize,
                                               apply_linear_trans=apply_linear_trans,
                                               scaling_bounds=scaling_bounds,
                                               rotation_bounds=rotation_bounds,
                                               shearing_bounds=shearing_bounds)
    training_generator = utils.build_training_generator(model_input_generator, batchsize)

    # prepare the segmentation model
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=augmentation_model)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                segmentation_label_list=label_list,
                                                input_model=wl2_model,
                                                metrics='wl2',
                                                name='metrics_model')
        if load_model_file is not None:
            print('loading ', load_model_file)
            wl2_model.load_weights(load_model_file)
        train_model(wl2_model, training_generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, log_dir,
                    'wl2', initial_epoch_wl2)

    # fine-tuning with dice metric
    if dice_epochs > 0:
        dice_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                 segmentation_label_list=label_list,
                                                 input_model=unet_model,
                                                 name='metrics_model')
        if wl2_epochs > 0:
            last_wl2_model_name = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)
            dice_model.load_weights(last_wl2_model_name, by_name=True)
        elif load_model_file is not None:
            print('loading ', load_model_file)
            dice_model.load_weights(load_model_file)
        train_model(dice_model, training_generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, log_dir,
                    'dice', initial_epoch_dice)


def labels_to_image_model(im_shape,
                          batchsize,
                          n_channels,
                          segmentation_labels,
                          n_neutral_labels,
                          output_shape=None,
                          output_div_by_n=None,
                          flipping=True,
                          aff=None,
                          apply_linear_trans=True,
                          apply_nonlin_trans=True,
                          nonlin_std=3.,
                          nonlin_shape_factor=.0625,
                          crop_channel2=None,
                          apply_bias_field=True,
                          bias_field_std=.3,
                          bias_shape_factor=.025):

    # reformat resolutions and get shapes
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    crop_shape = get_shapes(im_shape, output_shape, output_div_by_n)

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    new_segmentation_label_list, lut = utils.rearrange_label_list(segmentation_labels)

    # define model inputs
    image_input = KL.Input(shape=im_shape+[n_channels], name='image_input')
    labels_input = KL.Input(shape=im_shape + [1], name='labels_input')
    list_inputs = [image_input, labels_input]
    if apply_linear_trans:
        aff_in = KL.Input(shape=(n_dims + 1, n_dims + 1), name='aff_input')
        list_inputs.append(aff_in)
    else:
        aff_in = None

    # convert labels to new_label_list
    labels = l2i_et.convert_labels(labels_input, lut)

    # deform labels
    trans_inputs = list()
    if apply_linear_trans | apply_nonlin_trans:
        labels._keras_shape = tuple(labels.get_shape().as_list())
        labels = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(labels)
        if apply_linear_trans:
            trans_inputs.append(aff_in)
        if apply_nonlin_trans:
            small_shape = utils.get_resample_shape(im_shape, nonlin_shape_factor, n_dims)
            nonlin_shape = [batchsize] + small_shape
            nonlin_std_prior = KL.Lambda(lambda x: tf.random.uniform((1, 1), maxval=nonlin_std))([])
            elastic_trans = KL.Lambda(lambda x: tf.random.normal(nonlin_shape, stddev=x))(nonlin_std_prior)
            elastic_trans._keras_shape = tuple(elastic_trans.get_shape().as_list())
            resize_shape = [max(int(im_shape[i]/2), small_shape[i]) for i in range(n_dims)]
            nonlin_field = nrn_layers.Resize(size=resize_shape, interp_method='linear')(elastic_trans)
            nonlin_field = nrn_layers.VecInt()(nonlin_field)
            nonlin_field = nrn_layers.Resize(size=im_shape, interp_method='linear')(nonlin_field)
            trans_inputs.append(nonlin_field)
        labels = nrn_layers.SpatialTransformer(interp_method='nearest')([labels] + trans_inputs)
        labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)
        image = nrn_layers.SpatialTransformer(interp_method='linear')([image_input] + trans_inputs)
    else:
        image = image_input

    # crop labels
    if crop_shape != im_shape:
        labels, crop_idx = l2i_sa.random_cropping(labels, crop_shape, n_dims)
        image = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'),
                          size=tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32')))([image, crop_idx])

    # flip labels
    ras_axes = edit_volumes.get_ras_axes(aff, n_dims)
    flip_axis = [ras_axes[0] + 1]
    if flipping:
        assert aff is not None, 'aff should not be None if flipping is True'
        labels, flip = l2i_sa.label_map_random_flipping(labels, new_segmentation_label_list, n_neutral_labels, aff,
                                                        n_dims)
        image = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                             KL.Lambda(lambda x: K.reverse(x, axes=flip_axis))(y[1]),
                                             y[1]))([flip, image])

    # loop over channels
    if n_channels > 1:
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image)
    else:
        split = [image]

    processed_channels = list()
    for i, channel in enumerate(split):

        # reset edges of second channels to zero
        if (crop_channel2 is not None) & (i == 1):  # randomly crop sides of second channel
            crop_channel2 = utils.load_array_if_path(crop_channel2)
            channel, _ = l2i_sa.restrict_tensor(channel, axes=3, boundaries=crop_channel2)

        # apply bias field
        if apply_bias_field:
            channel = l2i_ia.bias_field_augmentation(channel, bias_field_std, bias_shape_factor)

        # intensity augmentation
        channel = l2i_ia.min_max_normalisation(channel)
        processed_channels.append(l2i_ia.gamma_augmentation(channel, std=0.5))

    # concatenate all channels back
    if n_channels > 1:
        image = KL.concatenate(processed_channels)
    else:
        image = processed_channels[0]

    # convert labels back to original values and reset unwanted labels to zero
    labels = l2i_et.convert_labels(labels, segmentation_labels)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'), name='labels_out')(labels)

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])
    brain_model = Model(inputs=list_inputs, outputs=[image, labels])

    return brain_model


def build_model_inputs(path_images,
                       path_label_maps,
                       batchsize=1,
                       apply_linear_trans=True,
                       scaling_bounds=None,
                       rotation_bounds=None,
                       shearing_bounds=None):

    # get label info
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_images[0])

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_label_maps), size=batchsize)

        # initialise input lists
        list_images = list()
        list_label_maps = list()
        list_affine_transforms = list()

        for idx in indices:

            # add image
            image = utils.load_volume(path_images[idx], aff_ref=np.eye(4))
            if n_channels > 1:
                list_images.append(utils.add_axis(image, axis=0))
            else:
                list_images.append(utils.add_axis(image, axis=-2))

            # add labels
            labels = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(labels, axis=-2))

            # add linear transform to inputs
            if apply_linear_trans:
                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = utils.draw_value_from_distribution(scaling_bounds, size=n_dims, centre=1, default_range=.15)
                if n_dims == 2:
                    rotation = utils.draw_value_from_distribution(rotation_bounds, default_range=15.0)
                else:
                    rotation = utils.draw_value_from_distribution(rotation_bounds, size=n_dims, default_range=15.0)
                shearing = utils.draw_value_from_distribution(shearing_bounds, size=n_dims**2-n_dims, default_range=.01)
                affine_transform = utils.create_affine_transformation_matrix(n_dims, scaling, rotation, shearing)
                list_affine_transforms.append(utils.add_axis(affine_transform))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if apply_linear_trans:
            list_inputs.append(list_affine_transforms)

        # concatenate individual input types if batchsize > 1
        if batchsize > 1:
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs


def get_shapes(im_shape, crop_shape, output_div_by_n):

    # reformat resolutions to lists
    im_shape = utils.reformat_to_list(im_shape)
    n_dims = len(im_shape)

    # crop_shape specified
    if crop_shape is not None:
        crop_shape = utils.reformat_to_list(crop_shape, length=n_dims, dtype='int')

        # make sure that crop_shape is smaller or equal to label shape
        crop_shape = [min(im_shape[i], crop_shape[i]) for i in range(n_dims)]

        # make sure crop_shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                         for s in crop_shape]
            if crop_shape != tmp_shape:
                print('crop_shape {0} not divisible by {1}, changed to {2}'.format(crop_shape, output_div_by_n,
                                                                                   tmp_shape))
                crop_shape = tmp_shape

    # no crop_shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            crop_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                          for s in im_shape]

        # if no need to be divisible by n, simply take labels_shape
        else:
            crop_shape = im_shape

    return crop_shape
