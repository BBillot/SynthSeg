import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model
import keras.activations as KA

from ext.lab2im import utils
from ext.lab2im.edit_tensors import pad_tensor


def sample_lesion(tensor, output_shape, mode='swiss', batchsize=1, path_prior=None):

    # get mask of potential areas with lesions (cerebral WM, cerebellar, WM, brainstem)
    mask = KL.Lambda(lambda x: tf.logical_or(tf.logical_or(tf.logical_or(tf.logical_or(K.equal(x, 2),
                                             K.equal(x, 7)), K.equal(x, 16)), K.equal(x, 41)), K.equal(x, 46)))(tensor)

    # sample code
    if mode == 'swiss':
        last_tensor = KL.Lambda(lambda x: tf.random.normal([batchsize, 5, 7, 5, 16], stddev=0.5))(tensor)
    elif mode == 'challenge':
        last_tensor = KL.Lambda(lambda x: tf.random.normal([batchsize, 12, 16, 12, 16], stddev=0.5))(tensor)
    elif mode == 'challenge_new':
        last_tensor = KL.Lambda(lambda x: tf.random.normal([batchsize, 9, 15, 9, 16], stddev=0.5))(tensor)
    else:
        raise Exception('mode should be swiss, challenge, or challenge_new, had: %s' % mode)

    # 1st deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, trainable=False, name='adeconv_0')(last_tensor)
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_0')(last_tensor)

    # 2nd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, activation='relu', trainable=False,
                                     name='adeconv_1')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_1')(last_tensor)

    # 3rd deconv layer
    if mode == 'swiss':
        last_tensor = KL.Conv3DTranspose(16, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                         name='adeconv_2')(last_tensor)
    else:
        last_tensor = KL.Conv3DTranspose(16, [5] * 3, activation='relu', trainable=False,
                                         name='adeconv_2')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_2')(last_tensor)

    # 4th deconv layer
    last_tensor = KL.Conv3DTranspose(24, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                     name='adeconv_3')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_3')(last_tensor)

    # 5th deconv layer
    last_tensor = KL.Conv3DTranspose(32, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                     name='adeconv_4')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_4')(last_tensor)

    # 6th deconv layer
    last_tensor = KL.Conv3DTranspose(1, [5] * 3, strides=[2] * 3, trainable=False, name='adeconv_5')(last_tensor)
    last_tensor = pad_tensor(last_tensor, [1] + output_shape + [1], pad_value=-100)
    last_tensor = KL.Lambda(lambda x: KA.sigmoid(x))(last_tensor)
    last_tensor = KL.Lambda(lambda x: tf.cast(x, 'float32'), name='lesion_map')(last_tensor)

    # multiply by prior lesion probability map and define threshold
    if path_prior is not None:
        lesion_prior = np.power(utils.load_volume(path_prior), 4)
        lesion_prior = np.tile(utils.add_axis(lesion_prior, axis=-2), [batchsize, 1, 1, 1, 1])
        prior = KL.Lambda(lambda x: tf.convert_to_tensor(lesion_prior))([])
        last_tensor = KL.multiply([last_tensor, prior])
        threshold = KL.Lambda(lambda x: tf.math.pow(tf.convert_to_tensor(10.), -tf.random.uniform((1, 1), 6, 8)))([])
    else:
        threshold = KL.Lambda(lambda x: tf.math.pow(tf.convert_to_tensor(10.), -tf.random.uniform((1, 1), 1, 4.5)))([])

    # add lesion to input tensor (only in regions specified by the computed mask)
    mask_lesion = KL.Lambda(lambda x: K.greater(x[0], x[1]), name='lesion')([last_tensor, threshold])
    mask = KL.Lambda(lambda x: tf.logical_and(x[0], x[1]), name='lesion_masked')([mask_lesion, mask])
    last_tensor = KL.Lambda(lambda x: tf.where(x[0], 77 * tf.ones_like(x[1]), tensor))([mask, tensor])

    return last_tensor


def build_decoder(code_shape, output_shape, mode='swiss', path_prior=None, batchsize=1):

    # get input
    inputs = KL.Input(code_shape)

    # 1st deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, trainable=False, name='adeconv_0')(inputs)
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_0')(last_tensor)

    # 2nd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, activation='relu', trainable=False, name='adeconv_1')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_1')(last_tensor)

    # 3rd deconv layer
    if mode == 'swiss':
        last_tensor = KL.Conv3DTranspose(16, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                         name='adeconv_2')(last_tensor)
    else:
        last_tensor = KL.Conv3DTranspose(16, [5] * 3, activation='relu', trainable=False,
                                         name='adeconv_2')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_2')(last_tensor)

    # 4th deconv layer
    last_tensor = KL.Conv3DTranspose(24, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                     name='adeconv_3')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_3')(last_tensor)

    # 5th deconv layer
    last_tensor = KL.Conv3DTranspose(32, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
                                     name='adeconv_4')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_4')(last_tensor)

    # 6th deconv layer
    last_tensor = KL.Conv3DTranspose(1, [5] * 3, strides=[2] * 3, trainable=False, name='adeconv_5')(last_tensor)
    last_tensor = KL.Lambda(lambda x: KA.sigmoid(x))(last_tensor)
    last_tensor = pad_tensor(last_tensor, [1] + output_shape + [1], pad_value=0)

    # multiply by prior lesion probability map and define threshold
    if path_prior is not None:
        lesion_prior = np.power(utils.load_volume(path_prior), 4)
        lesion_prior = np.tile(utils.add_axis(lesion_prior, axis=-2), [batchsize, 1, 1, 1, 1])
        prior = KL.Lambda(lambda x: tf.convert_to_tensor(lesion_prior))([])
        last_tensor = KL.multiply([last_tensor, prior])

    # create model
    decoder = Model(inputs=inputs, outputs=last_tensor)

    return decoder
