# python imports
import os
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import numpy.random as npr
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# project imports
from SynthSeg import metrics_model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models
from ext.neuron import layers as nrn_layers
from ext.lab2im import spatial_augmentation as l2i_sa
from ext.lab2im import intensity_augmentation as l2i_ia


def training(image_dir,
             labels_dir,
             cropping=None,
             flipping=True,
             scaling_range=0.07,
             rotation_range=10,
             shearing_range=0.01,
             nonlin_std_dev=3,
             nonlin_shape_fact=0.04,
             crop_channel_2=None,
             conv_size=3,
             n_levels=5,
             nb_conv_per_level=2,
             feat_multiplier=2,
             dropout=0,
             unet_feat_count=24,
             no_batch_norm=False,
             lr=1e-4,
             lr_decay=0,
             batch_size=1,
             wl2_epochs=50,
             dice_epochs=500,
             steps_per_epoch=100,
             background_weight=1e-4,
             include_background=False,
             load_model_file=None,
             initial_epoch_wl2=0,
             initial_epoch_dice=0,
             path_label_list=None,
             model_dir=None):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # prepare data files
    image_paths = utils.list_images_in_folder(image_dir)
    labels_paths = utils.list_images_in_folder(labels_dir)
    assert len(image_paths) == len(labels_paths), "There should be as many images as label maps."

    # get label and classes lists
    rotation_range = utils.load_array_if_path(rotation_range)
    scaling_range = utils.load_array_if_path(scaling_range)
    crop_channel_2 = utils.load_array_if_path(crop_channel_2)
    label_list, n_neutral_labels = utils.get_list_labels(label_list=path_label_list, FS_sort=True)
    n_labels = np.size(label_list)

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # create augmentation model and input generator
    im_shape, aff, _, n_channels, _, _ = utils.get_volume_info(image_paths[0])
    augmentation_model, unet_input_shape = labels_to_image_model(im_shape=im_shape,
                                                                 n_channels=n_channels,
                                                                 crop_shape=cropping,
                                                                 label_list=label_list,
                                                                 n_neutral_labels=n_neutral_labels,
                                                                 vox2ras=aff,
                                                                 nonlin_shape_factor=nonlin_shape_fact,
                                                                 crop_channel2=crop_channel_2,
                                                                 output_div_by_n=2 ** n_levels,
                                                                 flipping=flipping)

    model_input_generator = build_model_input_generator(images_paths=image_paths,
                                                        labels_paths=labels_paths,
                                                        n_channels=n_channels,
                                                        im_shape=im_shape,
                                                        scaling_range=scaling_range,
                                                        rotation_range=rotation_range,
                                                        shearing_range=shearing_range,
                                                        nonlin_shape_fact=nonlin_shape_fact,
                                                        nonlin_std_dev=nonlin_std_dev,
                                                        batch_size=batch_size)
    training_generator = utils.build_training_generator(model_input_generator, batch_size)

    # prepare the segmentation model
    if no_batch_norm:
        batch_norm_dim = None
    else:
        batch_norm_dim = -1
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 dilation_rate_mult=1,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=batch_norm_dim,
                                 input_model=augmentation_model)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                segmentation_label_list=label_list,
                                                input_model=wl2_model,
                                                metrics='weighted_l2',
                                                weight_background=background_weight,
                                                name='metrics_model')
        if load_model_file is not None:
            print('loading', load_model_file)
            wl2_model.load_weights(load_model_file)
        train_model(wl2_model, training_generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, log_dir,
                    'wl2', initial_epoch_wl2)

    # fine-tuning with dice metric
    if dice_epochs > 0:
        dice_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                 segmentation_label_list=label_list,
                                                 input_model=unet_model,
                                                 include_background=include_background,
                                                 name='metrics_model')
        if wl2_epochs > 0:
            last_wl2_model_name = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)
            dice_model.load_weights(last_wl2_model_name, by_name=True)
        elif load_model_file is not None:
            print('loading', load_model_file)
            dice_model.load_weights(load_model_file)
        train_model(dice_model, training_generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, log_dir,
                    'dice', initial_epoch_dice)


def labels_to_image_model(im_shape,
                          n_channels,
                          crop_shape,
                          label_list,
                          n_neutral_labels,
                          vox2ras,
                          nonlin_shape_factor=0.0625,
                          crop_channel2=None,
                          output_div_by_n=None,
                          flipping=True):

    # get shapes
    n_dims, _ = utils.get_dims(im_shape)
    crop_shape = get_shapes(crop_shape, im_shape, output_div_by_n)
    deformation_field_size = utils.get_resample_shape(im_shape, nonlin_shape_factor, len(im_shape))

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    new_label_list, lut = utils.rearrange_label_list(label_list)

    # define mandatory inputs
    image_input = KL.Input(shape=im_shape+[n_channels], name='image_input')
    labels_input = KL.Input(shape=im_shape + [1], name='labels_input')
    aff_in = KL.Input(shape=(n_dims + 1, n_dims + 1), name='aff_input')
    nonlin_field_in = KL.Input(shape=deformation_field_size, name='nonlin_input')
    list_inputs = [image_input, labels_input, aff_in, nonlin_field_in]

    # convert labels to new_label_list
    labels = KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(lut, dtype='int32'),
                                           tf.cast(x, dtype='int32')))(labels_input)

    # deform labels
    image_input._keras_shape = tuple(image_input.get_shape().as_list())
    labels._keras_shape = tuple(labels.get_shape().as_list())
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='float'))(labels)
    resize_shape = [max(int(im_shape[i] / 2), deformation_field_size[i]) for i in range(len(im_shape))]
    nonlin_field = nrn_layers.Resize(size=resize_shape, interp_method='linear')(nonlin_field_in)
    nonlin_field = nrn_layers.VecInt()(nonlin_field)
    nonlin_field = nrn_layers.Resize(size=im_shape, interp_method='linear')(nonlin_field)
    image = nrn_layers.SpatialTransformer(interp_method='linear')([image_input, aff_in, nonlin_field])
    labels = nrn_layers.SpatialTransformer(interp_method='nearest')([labels, aff_in, nonlin_field])
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)

    # cropping
    if crop_shape is not None:
        image, crop_idx = l2i_sa.random_cropping(image, crop_shape, n_dims)
        labels = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'),
                           size=tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32')))([labels, crop_idx])
    else:
        crop_shape = im_shape

    # flipping
    if flipping:
        labels, flip = l2i_sa.label_map_random_flipping(labels, label_list, n_neutral_labels, vox2ras, n_dims)
        ras_axes, _ = edit_volumes.get_ras_axes_and_signs(vox2ras, n_dims)
        flip_axis = [ras_axes[0] + 1]
        image = KL.Lambda(lambda y: K.switch(y[0],
                                             KL.Lambda(lambda x: K.reverse(x, axes=flip_axis))(y[1]),
                                             y[1]))([flip, image])

    # convert labels back to original values
    labels = KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(label_list, dtype='int32'),
                                           tf.cast(x, dtype='int32')), name='labels_out')(labels)

    # intensity augmentation
    image = KL.Lambda(lambda x: K.clip(x, 0, 300), name='clipping')(image)

    # loop over channels
    if n_channels > 1:
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image)
    else:
        split = [image]
    processed_channels = list()
    for i, channel in enumerate(split):

        # normalise and shift intensities
        image = l2i_ia.min_max_normalisation(image)
        image = KL.Lambda(lambda x: K.random_uniform((1,), .85, 1.1) * x + K.random_uniform((1,), -.3, .3))(image)
        image = KL.Lambda(lambda x: K.clip(x, 0, 1))(image)
        image = l2i_ia.gamma_augmentation(image)

        # randomly crop sides of second channel
        if (crop_channel2 is not None) & (channel == 1):
            image = l2i_sa.restrict_tensor(image, crop_channel2, n_dims)

    # concatenate all channels back, and clip output (include labels to keep it when plugging to other models)
    if n_channels > 1:
        image = KL.concatenate(processed_channels)
    else:
        image = processed_channels[0]
    image = KL.Lambda(lambda x: K.clip(x[0], 0, 1), name='image_out')([image, labels])

    # build model
    brain_model = Model(inputs=list_inputs, outputs=[image, labels])
    # shape of returned images
    output_shape = image.get_shape().as_list()[1:]

    return brain_model, output_shape


def get_shapes(crop_shape, im_shape, div_by_n):
    n_dims, _ = utils.get_dims(im_shape)
    # crop_shape specified
    if crop_shape is not None:
        crop_shape = utils.reformat_to_list(crop_shape, length=n_dims, dtype='int')
        crop_shape = [min(im_shape[i], crop_shape[i]) for i in range(n_dims)]
        # make sure output shape is divisible by output_div_by_n
        if div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, div_by_n, smaller_ans=True) for s in crop_shape]
            if crop_shape != tmp_shape:
                print('crop shape {0} not divisible by {1}, changed to {2}'.format(crop_shape, div_by_n, tmp_shape))
                crop_shape = tmp_shape
    # no crop_shape, so no cropping unless image shape is not divisible by output_div_by_n
    else:
        if div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, div_by_n, smaller_ans=True) for s in im_shape]
            if tmp_shape != im_shape:
                print('image shape {0} not divisible by {1}, cropped to {2}'.format(im_shape, div_by_n, tmp_shape))
                crop_shape = tmp_shape
    return crop_shape


def build_model_input_generator(images_paths,
                                labels_paths,
                                n_channels,
                                im_shape,
                                scaling_range=None,
                                rotation_range=None,
                                shearing_range=None,
                                nonlin_shape_fact=0.0625,
                                nonlin_std_dev=3,
                                batch_size=1):

    # Generate!
    while True:

        # randomly pick as many images as batch_size
        indices = npr.randint(len(images_paths), size=batch_size)

        # initialise input tensors
        images_all = []
        labels_all = []
        aff_all = []
        nonlinear_field_all = []

        for idx in indices:

            # add image
            image = utils.load_volume(images_paths[idx])
            if n_channels > 1:
                images_all.append(utils.add_axis(image, axis=0))
            else:
                images_all.append(utils.add_axis(image, axis=-2))

            # add labels
            labels = utils.load_volume(labels_paths[idx], dtype='int')
            labels_all.append(utils.add_axis(labels, axis=-2))

            # get affine transformation: rotate, scale, shear (translation done during random cropping)
            n_dims, _ = utils.get_dims(im_shape)
            scaling = utils.draw_value_from_distribution(scaling_range, size=n_dims, centre=1, default_range=.15)
            if n_dims == 2:
                rotation_angle = utils.draw_value_from_distribution(rotation_range, default_range=15.0)
            else:
                rotation_angle = utils.draw_value_from_distribution(rotation_range, size=n_dims, default_range=15.0)
            shearing = utils.draw_value_from_distribution(shearing_range, size=n_dims ** 2 - n_dims, default_range=.01)
            aff = utils.create_affine_transformation_matrix(n_dims, scaling, rotation_angle, shearing)
            aff_all.append(utils.add_axis(aff))

            # add non linear field
            deform_shape = utils.get_resample_shape(im_shape, nonlin_shape_fact, len(im_shape))
            nonlinear_field = npr.normal(loc=0, scale=nonlin_std_dev * npr.rand(), size=deform_shape)
            nonlinear_field_all.append(utils.add_axis(nonlinear_field))

        # build list of inputs of the augmentation model
        inputs_vals = [images_all, labels_all, aff_all, nonlinear_field_all]

        # put images and labels (concatenated if batch_size>1) into a tuple of 2 elements: (cat_images, cat_labels)
        if batch_size > 1:
            inputs_vals = [np.concatenate(item, 0) for item in inputs_vals]
        else:
            inputs_vals = [item[0] for item in inputs_vals]

        yield inputs_vals


def train_model(model,
                generator,
                lr,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                log_dir,
                metric_type,
                initial_epoch=0):

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % metric_type)
    temp_callbacks = ModelCheckpoint(save_file_name, verbose=1)
    mg_model = model

    # TensorBoard callback
    if metric_type == 'dice':
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
        callbacks = [temp_callbacks, tensorboard]
    else:
        callbacks = [temp_callbacks]

    # metric and loss
    metric = metrics_model.IdentityLoss()
    data_loss = metric.loss

    # compile
    mg_model.compile(optimizer=Adam(lr=lr, decay=lr_decay), loss=data_loss, loss_weights=[1.0])

    # fit
    if metric_type == 'dice':
        mg_model.fit_generator(generator, epochs=n_epochs, steps_per_epoch=n_steps, callbacks=callbacks,
                               initial_epoch=initial_epoch)
    else:
        mg_model.fit_generator(generator, epochs=n_epochs, steps_per_epoch=n_steps, callbacks=callbacks)
