# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_tensors as l2i_et


def metrics_model(input_shape,
                  segmentation_label_list,
                  input_model=None,
                  loss_cropping=None,
                  metrics='dice',
                  name=None,
                  prefix=None,
                  validation_on_real_images=False):

    # naming the model
    model_name = name
    if prefix is None:
        prefix = model_name

    # first layer: input
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]
        last_tensor = KL.Reshape(input_shape, name='predicted_output')(last_tensor)

    # get deformed labels
    n_labels = input_shape[-1]
    if validation_on_real_images:
        labels_gt = KL.Input(shape=input_shape[:-1]+[1], name='labels_input')
        input_tensor = [input_tensor[0], labels_gt]
    else:
        labels_gt = input_model.get_layer('labels_out').output

    # convert gt labels to 0...N-1 values
    n_labels = segmentation_label_list.shape[0]
    _, lut = utils.rearrange_label_list(segmentation_label_list)
    labels_gt = l2i_et.convert_labels(labels_gt, lut)

    # convert gt labels to probabilistic values
    labels_gt = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels_gt)
    labels_gt = KL.Reshape(input_shape)(labels_gt)
    labels_gt = KL.Lambda(lambda x: K.clip(x / K.sum(x, axis=-1, keepdims=True), K.epsilon(), 1))(labels_gt)

    # crop output to evaluate loss function in centre patch
    if loss_cropping is not None:
        # format loss_cropping
        labels_shape = labels_gt.get_shape().as_list()[1:-1]
        n_dims, _ = utils.get_dims(labels_shape)
        loss_cropping = [-1] + utils.reformat_to_list(loss_cropping, length=n_dims) + [-1]
        # perform cropping
        begin_idx = [0] + [int((labels_shape[i] - loss_cropping[i]) / 2) for i in range(n_dims)] + [0]
        labels_gt = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor(begin_idx, dtype='int32'),
                              size=tf.convert_to_tensor(loss_cropping, dtype='int32')))(labels_gt)
        last_tensor = KL.Lambda(lambda x: tf.slice(x, begin=tf.convert_to_tensor(begin_idx, dtype='int32'),
                                size=tf.convert_to_tensor(loss_cropping, dtype='int32')))(last_tensor)

    # metrics is computed as part of the model
    if metrics == 'dice':

        # make sure predicted values are probabilistic
        last_tensor = KL.Lambda(lambda x: K.clip(x / K.sum(x, axis=-1, keepdims=True), K.epsilon(), 1))(last_tensor)

        # compute dice
        top = KL.Lambda(lambda x: 2*x[0]*x[1])([labels_gt, last_tensor])
        bottom = KL.Lambda(lambda x: K.square(x[0]) + K.square(x[1]))([labels_gt, last_tensor])
        for dims_to_sum in range(len(input_shape)-1):
            top = KL.Lambda(lambda x: K.sum(x, axis=1))(top)
            bottom = KL.Lambda(lambda x: K.sum(x, axis=1))(bottom)
        last_tensor = KL.Lambda(lambda x: x[0] / K.maximum(x[1], 0.001), name='dice')([top, bottom])  # 1d vector

        # compute mean dice loss
        w = np.ones([n_labels]) / n_labels
        w[0] = 0.0
        last_tensor = KL.Lambda(lambda x: 1 - x, name='dice_loss')(last_tensor)
        last_tensor = KL.Lambda(lambda x: K.sum(x*tf.convert_to_tensor(w, dtype='float32'), axis=1),
                                name='mean_dice_loss')(last_tensor)
        # average mean dice loss over mini batch
        last_tensor = KL.Lambda(lambda x: K.mean(x), name='average_mean_dice_loss')(last_tensor)

    elif metrics == 'wl2':
        # compute weighted l2 loss
        weights = KL.Lambda(lambda x: K.expand_dims(1 - x[..., 0] + 1e-4))(labels_gt)  # 1e-4 = background weight
        normaliser = KL.Lambda(lambda x: K.sum(x[0]) * K.int_shape(x[1])[-1])([weights, last_tensor])
        # last_tensor = KL.Lambda(lambda x: K.sum(x[2] * K.square(x[1] - 15 * (2 * x[0] - 1))) / x[3])([
        #                         labels_gt, last_tensor, weights, normaliser])
        last_tensor = KL.Lambda(lambda x: K.sum(x[2] * K.square(x[1] - 5 * (2 * x[0] - 1))) / x[3])([
                                labels_gt, last_tensor, weights, normaliser])

    else:
        raise Exception('metrics should either be "dice or "wl2, got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_tensor, outputs=last_tensor, name=model_name)
    return model


def evaluation_model(input_shape, segmentation_label_list):
    """model to compute hard dice scores from maps. Input shape designated width*height*depth*channels"""

    # get deformed labels
    labels_gt = KL.Input(shape=input_shape, name='gt_input')
    labels_seg = KL.Input(shape=input_shape, name='seg_input')

    # loop through all labels in label list
    dice = KL.Lambda(lambda x: tf.zeros([0]), name='empty_dice')(labels_gt)
    for label in segmentation_label_list:

        # get hard segmentation
        mask_gt = KL.Lambda(lambda x: tf.where(K.equal(x, label),
                                               tf.ones_like(x, dtype='float32'),
                                               tf.zeros_like(x, dtype='float32')),
                            name='gt_mask_%s' % label)(labels_gt)
        mask_seg = KL.Lambda(lambda x: tf.where(K.equal(x, label),
                                                tf.ones_like(x, dtype='float32'),
                                                tf.zeros_like(x, dtype='float32')),
                             name='seg_mask_%s' % label)(labels_seg)

        # compute dice
        top = KL.Lambda(lambda x: 2 * x[0] * x[1], name='top%s' % label)([mask_gt, mask_seg])
        bottom = KL.Lambda(lambda x: K.square(x[0]) + K.square(x[1]), name='bottom%s' % label)([mask_gt, mask_seg])
        for dims_to_sum in range(len(input_shape)):
            top = KL.Lambda(lambda x: K.sum(x, axis=1), name='top_sum_{0}_{1}'.format(label, dims_to_sum))(top)
            bottom = KL.Lambda(lambda x: K.sum(x, axis=1), name='bottom_sum_{0}_{1}'.format(label, dims_to_sum))(bottom)
        tmp_dice = KL.Lambda(lambda x: x[0] / K.maximum(x[1], 0.001), name='dice%s' % label)([top, bottom])  # 1d vector

        # concat to other values
        dice = KL.Lambda(lambda x: tf.concat(x, axis=0), name='cat_%s' % label)([dice, tmp_dice])
    dice = KL.Lambda(lambda x: tf.expand_dims(x, 0), name='add_dim')(dice)

    # create the model and return
    model = Model(inputs=[labels_gt, labels_seg], outputs=dice)
    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
