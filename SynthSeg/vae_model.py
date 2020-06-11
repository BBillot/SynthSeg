import keras.layers as KL
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
