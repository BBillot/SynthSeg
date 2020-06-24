import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model
import keras.activations as KA
from ext.lab2im.edit_tensors import pad_tensor


def build_decoder(shape, output_shape):

    # get input
    inputs = KL.Input(shape)

    # 1st deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3]*3, trainable=False, name='adeconv_0')(inputs)
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_0')(last_tensor)

    # 2nd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3]*3, activation='relu', trainable=False,
                                     name='adeconv_1')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_1')(last_tensor)

    # 3rd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [5]*3, strides=[2]*3, activation='relu', trainable=False,
                                     name='adeconv_2')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_2')(last_tensor)

    # 4th deconv layer
    last_tensor = KL.Conv3DTranspose(24, [5]*3, strides=[2]*3, activation='relu', trainable=False,
                                     name='adeconv_3')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_3')(last_tensor)

    # 5th deconv layer
    last_tensor = KL.Conv3DTranspose(32, [5]*3, strides=[2]*3, activation='relu', trainable=False,
                                     name='adeconv_4')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_4')(last_tensor)

    # 6th deconv layer
    last_tensor = KL.Conv3DTranspose(1, [5]*3, strides=[2]*3, trainable=False, name='adeconv_5')(last_tensor)
    last_tensor = pad_tensor(last_tensor, [1] + output_shape + [1], pad_value=-100)
    last_tensor = KL.Lambda(lambda x: KA.sigmoid(x))(last_tensor)

    # create model
    decoder = Model(inputs=inputs, outputs=last_tensor)

    return decoder


def sample_lesion(tensor, output_shape, batchsize=1):

    # first get mask of input label tensor (cerebral WM, cerebellar, WM, brainstem, and thalamus with 10%)
    mask = KL.Lambda(lambda x: tf.logical_or(tf.logical_or(tf.logical_or(tf.logical_or(K.equal(x, 2),
                                             K.equal(x, 7)), K.equal(x, 16)), K.equal(x, 41)), K.equal(x, 46)))(tensor)
    mask = KL.Lambda(lambda y: K.switch(tf.squeeze(K.greater(tf.random.uniform((1, 1), 0, 1), 0.9)),
                                        KL.Lambda(lambda x: tf.logical_or(tf.logical_or(
                                            x[0], K.equal(x[1], 10)), K.equal(x[1], 49)))([y[0], y[1]]),
                                        y[0]))([mask, tensor])

    # sample code
    last_tensor = KL.Lambda(lambda x: tf.random.normal([batchsize, 5, 7, 5, 16]))(tensor)

    # 1st deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, trainable=False, name='adeconv_0')(last_tensor)
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_0')(last_tensor)

    # 2nd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [3] * 3, activation='relu', trainable=False,
                                     name='adeconv_1')(last_tensor)
    last_tensor = KL.BatchNormalization(trainable=False, name='abatch_n_d_1')(last_tensor)

    # 3rd deconv layer
    last_tensor = KL.Conv3DTranspose(16, [5] * 3, strides=[2] * 3, activation='relu', trainable=False,
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

    # sample threshold value for lesion probability map
    threshold = KL.Lambda(lambda x: tf.random.uniform((1, 1), -1, 5))([])
    threshold = KL.Lambda(lambda x: tf.math.pow(tf.convert_to_tensor(10.), -x), name='threshold')(threshold)

    # add lesion to input tensor (ionly in regions specified by the computed mask)
    mask_lesion = KL.Lambda(lambda x: K.greater(x[0], x[1]))([last_tensor, threshold])
    mask = KL.Lambda(lambda x: tf.logical_and(x[0], x[1]))([mask_lesion, mask])
    last_tensor = KL.Lambda(lambda x: tf.where(x[0], 77 * tf.ones_like(x[1]), tensor), name='lesion')([mask, tensor])

    return last_tensor
