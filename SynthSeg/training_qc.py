"""

This function trains a regressor network to predict Dice scores between segmentations (typically obtained with an
automated algorithm), and their ground truth (to which wwe typically do not have access at test time).

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
import keras
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL
import keras.backend as K
import keras.callbacks as KC
from keras.optimizers import Adam
from inspect import getmembers, isclass

# project imports
from SynthSeg import metrics_model as metrics

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers as l2i_layers
from ext.neuron import utils as nrn_utils
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models


def training(list_paths_input_labels,
             list_paths_target_labels,
             model_dir,
             labels_list,
             labels_list_to_convert=None,
             subjects_prob=None,
             batchsize=1,
             output_shape=None,
             scaling_bounds=.2,
             rotation_bounds=15,
             shearing_bounds=.012,
             translation_bounds=False,
             nonlin_std=4.,
             nonlin_scale=.04,
             n_levels=5,
             nb_conv_per_level=3,
             conv_size=5,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='relu',
             lr=1e-4,
             epochs=300,
             steps_per_epoch=1000,
             checkpoint=None):

    """
    This function trains a regressor network to predict Dice scores between segmentations (typically obtained with an
    automated algorithm), and their ground truth (to which we typically do not have access at test time).

    # IMPORTANT !!!
    # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
    # these values refer to the RAS axes.

    :param list_paths_input_labels: list of all the paths of the input label maps. These typically correspond to the
    outputs given by a segmentation algorithm, for which the network will try to predict quality scores.
    :param list_paths_target_labels: list of all the paths of the ground truths for the input label maps. These are used
    to compute Dice scores with their corresponding input label maps, that will serve as target for the regression
    network. As such, target label maps must have the same label values as the input label maps.
    :param model_dir: path of a directory where the models will be saved during training.
    :param labels_list: list of all the label values present in the input/GT label maps.

    #---------------------------------------------- Generation parameters ----------------------------------------------
    # label maps parameters
    :param labels_list_to_convert: (optional) instead of regressing Dice scores for all the labels present in the
    input/GT label maps, one can "regroup" them into "groups". This must be a sequence or a 1d numpy array of the same
    length as labels_list with the corresponding group. By default (None), regression scores are computed for all labels
    :param subjects_prob: (optional) relative order of importance (doesn't have to be probabilistic), with which to pick
    the provided label maps at each minibatch. Can be a sequence, a 1D numpy array, or the path to such an array, and it
    must be as long as path_label_maps. By default, all label maps are chosen with the same importance.

    # output-related parameters
    :param batchsize: (optional) number of images to generate per mini-batch. Default is 1.
    :param output_shape: (optional) desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.

    # spatial deformation parameters
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

    # ------------------------------------------ UNet architecture parameters ------------------------------------------
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature for the first layer of the UNet. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this number at each new level. Default is 2.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.

    # ----------------------------------------------- Training parameters ----------------------------------------------
    :param lr: (optional) learning rate for the training. Default is 1e-4
    :param epochs: (optional) number of training epochs. Default is 300.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 10000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param checkpoint: (optional) path of an already saved model to load before starting the training.
    """

    # prepare data files
    labels_list, _ = utils.get_list_labels(label_list=labels_list)
    if labels_list_to_convert is not None:
        labels_list_to_convert, _ = utils.get_list_labels(label_list=labels_list_to_convert)
    n_labels = len(np.unique(labels_list))

    # create augmentation model
    labels_shape, _, n_dims, _, _, _ = utils.get_volume_info(list_paths_target_labels[0], aff_ref=np.eye(4))
    augmentation_model = build_augmentation_model(labels_shape,
                                                  labels_list,
                                                  labels_list_to_convert=labels_list_to_convert,
                                                  output_shape=output_shape,
                                                  output_div_by_n=2 ** n_levels,
                                                  scaling_bounds=scaling_bounds,
                                                  rotation_bounds=rotation_bounds,
                                                  shearing_bounds=shearing_bounds,
                                                  translation_bounds=translation_bounds,
                                                  nonlin_std=nonlin_std,
                                                  nonlin_scale=nonlin_scale)

    # prepare QC model
    regression_model = build_qc_model(input_model=augmentation_model,
                                      n_labels=n_labels,
                                      n_levels=n_levels,
                                      nb_conv_per_level=nb_conv_per_level,
                                      conv_size=conv_size,
                                      unet_feat_count=unet_feat_count,
                                      feat_multiplier=feat_multiplier,
                                      activation=activation)
    qc_model = build_qc_loss(regression_model)

    # input generator
    model_inputs = build_model_inputs(path_input_label_maps=list_paths_input_labels,
                                      path_target_label_maps=list_paths_target_labels,
                                      batchsize=batchsize,
                                      subjects_prob=subjects_prob)
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    train_model(qc_model, input_generator, lr, epochs, steps_per_epoch, model_dir, 'qc', checkpoint)


def build_augmentation_model(labels_shape,
                             labels_list,
                             labels_list_to_convert=None,
                             output_shape=None,
                             output_div_by_n=None,
                             scaling_bounds=0.15,
                             rotation_bounds=15,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=3.,
                             nonlin_scale=.0625):

    # reformat resolutions and get shapes
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)

    # get shapes
    output_shape = get_shapes(labels_shape, output_shape, output_div_by_n, n_dims)

    # define model inputs
    net_input = KL.Input(shape=labels_shape + [1], name='noisy_labels_input', dtype='int32')
    target_input = KL.Input(shape=labels_shape + [1], name='target_input', dtype='int32')

    # convert labels if necessary
    if labels_list_to_convert is not None:
        noisy_labels = l2i_layers.ConvertLabels(labels_list_to_convert, labels_list, name='convert_noisy')(net_input)
        target = l2i_layers.ConvertLabels(labels_list_to_convert, labels_list, name='convert_target')(target_input)
    else:
        noisy_labels = net_input
        target = target_input

    # deform labels
    noisy_labels, target = l2i_layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                               rotation_bounds=rotation_bounds,
                                                               shearing_bounds=shearing_bounds,
                                                               translation_bounds=translation_bounds,
                                                               nonlin_std=nonlin_std,
                                                               nonlin_scale=nonlin_scale,
                                                               inter_method='nearest')([noisy_labels, target])

    # mask image, compute Dice score with full GT, and crop noisy labels
    noisy_labels, scores = SimulatePartialFOV(crop_shape=output_shape[0],
                                              labels_list=np.unique(labels_list),
                                              min_fov_shape=70,
                                              prob_mask=0.3, name='partial_fov')([noisy_labels, target])

    # dummy layers
    scores = KL.Lambda(lambda x: x, name='dice_gt')(scores)
    noisy_labels = KL.Lambda(lambda x: x[0], name='labels_out')([noisy_labels, scores])

    # build model and return
    brain_model = models.Model(inputs=[net_input, target_input], outputs=noisy_labels)
    return brain_model


def build_qc_model(input_model,
                   n_labels,
                   n_levels,
                   nb_conv_per_level,
                   conv_size,
                   unet_feat_count,
                   feat_multiplier,
                   activation):

    # get prediction
    last_tensor = input_model.outputs[0]
    input_shape = last_tensor.get_shape().as_list()[1:]
    assert input_shape[-1] == n_labels, 'mismatch between number of predicted labels, and segmentation labels'

    # build model
    model = nrn_models.conv_enc(input_model=input_model,
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

    return models.Model(input_model.inputs, last)


def build_qc_loss(input_model):

    # get Dice scores
    dice_gt = input_model.get_layer('dice_gt').output
    dice_pred = input_model.outputs[0]

    # get loss
    loss = KL.Lambda(lambda x: K.sum(K.mean(K.square(x[0] - x[1]), axis=0)), name='qc_loss')([dice_gt, dice_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())

    return models.Model(inputs=input_model.inputs, outputs=loss)


def build_model_inputs(path_input_label_maps,
                       path_target_label_maps,
                       batchsize=1,
                       subjects_prob=None):

    # make sure subjects_prob sums to 1
    subjects_prob = utils.load_array_if_path(subjects_prob)
    if subjects_prob is not None:
        subjects_prob /= np.sum(subjects_prob)

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = np.random.choice(np.arange(len(path_input_label_maps)), size=batchsize, p=subjects_prob)

        # initialise input lists
        list_input_label_maps = list()
        list_target_label_maps = list()

        for idx in indices:

            # load input
            input_net = utils.load_volume(path_input_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_input_label_maps.append(utils.add_axis(input_net, axis=[0, -1]))

            # load target
            target = utils.load_volume(path_target_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_target_label_maps.append(utils.add_axis(target, axis=[0, -1]))

        # build list of training pairs
        list_training_pairs = [list_input_label_maps, list_target_label_maps]
        if batchsize > 1:  # concatenate individual input types if batchsize > 1
            list_training_pairs = [np.concatenate(item, 0) for item in list_training_pairs]
        else:
            list_training_pairs = [item[0] for item in list_training_pairs]

        yield list_training_pairs


def get_shapes(labels_shape, cropping_shape, output_div_by_n, n_dims):

    # cropping shape specified, make sure it's okay
    if cropping_shape is not None:
        cropping_shape = utils.reformat_to_list(cropping_shape, length=n_dims, dtype='int')

        # make sure that cropping shape is smaller or equal to label shape
        cropping_shape = [min(labels_shape[i], cropping_shape[i]) for i in range(n_dims)]

        # make sure cropping shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in cropping_shape]
            if cropping_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.format(cropping_shape, output_div_by_n,
                                                                                     tmp_shape))
                cropping_shape = tmp_shape

    # no cropping shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:

        # make sure labels shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            cropping_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in labels_shape]

        # if no need to be divisible by n, simply take cropping_shape as image_shape, and build output_shape
        else:
            cropping_shape = labels_shape

    return cropping_shape


def train_model(model,
                generator,
                learning_rate,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                path_checkpoint=None,
                reinitialise_momentum=False):

    # prepare model and log folders
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, 'qc_{epoch:03d}.h5')
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1),
                 KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)]

    compile_model = True
    init_epoch = 0
    if (path_checkpoint is not None) & (not reinitialise_momentum):
        init_epoch = int(os.path.basename(path_checkpoint).split(metric_type)[1][1:-3])
        custom_l2i = {key: value for (key, value) in getmembers(l2i_layers, isclass) if key != 'Layer'}
        custom_nrn = {key: value for (key, value) in getmembers(nrn_layers, isclass) if key != 'Layer'}
        custom_objects = {**custom_l2i, **custom_nrn, 'tf': tf, 'keras': keras, 'loss': metrics.IdentityLoss().loss}
        model = models.load_model(path_checkpoint, custom_objects=custom_objects)
        compile_model = False
    elif path_checkpoint is not None:
        model.load_weights(path_checkpoint, by_name=True)

    # compile
    if compile_model:
        model.compile(optimizer=Adam(lr=learning_rate), loss=metrics.IdentityLoss().loss)

    # fit
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=init_epoch)


class SimulatePartialFOV(KL.Layer):
    """Expects hard segmentations for the two input label maps, input first, gt second."""

    def __init__(self, crop_shape, labels_list, min_fov_shape, prob_mask, **kwargs):

        # cropping arguments
        self.crop_shape = crop_shape
        self.crop_max_val = None

        # asking arguments
        self.min_fov_shape = min_fov_shape
        self.prob_mask = prob_mask

        # shape arguments
        self.inshape = None
        self.n_dims = None
        self.labels_list = labels_list
        self.n_labels = len(labels_list)
        self.lut = None
        self.meshgrid = None

        super(SimulatePartialFOV, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["labels_list"] = self.labels_list
        config["crop_shape"] = self.crop_shape
        config["min_fov_shape"] = self.min_fov_shape
        config["prob_mask"] = self.prob_mask
        return config

    def build(self, input_shape):

        # check shapes
        assert len(input_shape) == 2, 'SimulatePartialFOV expects 2 inputs: labels to deform and GT (for Dice scores).'
        assert input_shape[0] == input_shape[1], 'the two inputs must have the same shape.'
        self.n_dims = len(input_shape[0]) - 2
        self.inshape = input_shape[0][1:self.n_dims + 1]

        self.crop_max_val = self.inshape[0] - self.crop_shape
        self.meshgrid = nrn_utils.volshape_to_ndgrid(self.inshape)
        self.lut = tf.convert_to_tensor(utils.get_mapping_lut(self.labels_list), dtype='int32')

        self.built = True
        super(SimulatePartialFOV, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # get inputs
        x = inputs[0][..., 0]
        y = inputs[1][..., 0]

        # sample cropping indices
        batchsize = tf.split(tf.shape(x), [1, -1])[0]
        sample_shape = tf.concat([batchsize, self.n_dims * tf.ones([1], dtype='int32')], 0)
        if self.crop_max_val > 0:
            crop_idx_inf = tf.random.uniform(shape=sample_shape, minval=0, maxval=self.crop_max_val, dtype='int32')
            crop_idx_sup = crop_idx_inf + self.crop_shape
        else:
            crop_idx_inf = crop_idx_sup = None

        # sample masking indices
        fov_shape = tf.random.uniform(sample_shape, minval=self.min_fov_shape, maxval=self.inshape[0], dtype='int32')
        mask_idx_inf = tf.random.uniform(shape=sample_shape, minval=0, maxval=1, dtype='float32')
        mask_idx_inf_tmp = tf.cast(mask_idx_inf * tf.cast(self.inshape[0]-fov_shape, 'float32'), 'int32')
        mask_idx_sup_tmp = mask_idx_inf_tmp + fov_shape
        if self.crop_max_val > 0:
            mask_idx_inf = tf.maximum(mask_idx_inf_tmp, crop_idx_inf)
            mask_idx_sup = tf.minimum(mask_idx_sup_tmp, crop_idx_sup)
        else:
            mask_idx_inf = mask_idx_inf_tmp
            mask_idx_sup = mask_idx_sup_tmp

        # mask input labels
        mask = tf.map_fn(self._single_build_mask, [x, mask_idx_inf, mask_idx_sup], tf.int32)
        x = K.switch(tf.squeeze(K.greater(tf.random.uniform([1], 0, 1), 1 - self.prob_mask)), x * mask, x)

        # compute dice score for each label value
        x = tf.one_hot(tf.gather(self.lut, x), depth=self.n_labels, axis=-1)
        y = tf.one_hot(tf.gather(self.lut, y), depth=self.n_labels, axis=-1)
        top = tf.math.reduce_sum(2 * x * y, axis=list(range(1, self.n_dims + 1)))
        bottom = x + y + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, self.n_dims + 1)))
        dice_score = top / bottom

        # crop input labels
        if self.crop_max_val > 0:
            x_cropped = tf.map_fn(self._single_slice, [x, crop_idx_inf], dtype=tf.float32)
        else:
            x_cropped = x

        return [x_cropped, dice_score]

    def _single_build_mask(self, inputs):
        vol = inputs[0]
        mask_idx_inf = inputs[1]
        mask_idx_sup = inputs[2]
        mask = tf.ones(vol.shape, dtype='bool')
        for i in range(self.n_dims):
            tmp_mask_inf = tf.less(self.meshgrid[i], mask_idx_inf[i])
            tmp_mask_sup = tf.greater(self.meshgrid[i], mask_idx_sup[i])
            mask = tf.logical_and(mask, tf.logical_not(tf.logical_or(tmp_mask_inf, tmp_mask_sup)))
        return tf.cast(mask, 'int32')

    def _single_slice(self, inputs):
        vol = inputs[0]
        crop_idx_inf = inputs[1]
        crop_idx_inf = tf.concat([tf.cast(crop_idx_inf, 'int32'), tf.zeros([1], dtype='int32')], axis=0)
        crop_size = tf.convert_to_tensor([self.crop_shape] * self.n_dims + [-1], dtype='int32')
        return tf.slice(vol, begin=crop_idx_inf, size=crop_size)

    def compute_output_shape(self, input_shape):
        return [(None, *[self.crop_shape] * self.n_dims, self.n_labels), (None, self.n_labels)]
