import tensorflow as tf
import keras.layers as KL
import keras.backend as K


def sample_gmm_conditioned_on_labels(labels, means, std_devs, n_labels, n_channels):
    """This function generates an image tensor by sampling a Gaussian Mixture Model conditioned on a label map.
    The generated image can be multi-spectral.
    :param labels: input label map tensor with a batch size of N
    :param means: means of the GMM per channel, should have shape [N, n_labels, n_channels]
    :param std_devs: std devs of the GMM per channel, should have shape [N, n_labels, n_channels]
    :param n_labels: number of labels in the input tensor label map
    :param n_channels: number of channels to generate
    :return: image tensor of shape [N, ..., n_channels]
    """

    # sample from normal distribution
    image = KL.Lambda(lambda x: tf.random.normal(tf.shape(x)))(labels)

    # one channel
    if n_channels == 1:
        means = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels])))(means)
        means_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([means, labels])

        std_devs = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels])))(std_devs)
        std_devs_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([std_devs, labels])

    # multi-channel
    else:
        cat_labels = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32') + n_labels*i
                                                    for i in range(n_channels)], -1))(labels)

        means = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels, n_channels])))(means)
        means = KL.Lambda(lambda x: K.reshape(tf.concat(tf.split(x, [1]*n_channels, axis=-1), 0),
                                              tuple([n_labels*n_channels])))(means)
        means_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([means, cat_labels])

        std_devs = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels, n_channels])))(std_devs)
        std_devs = KL.Lambda(lambda x: K.reshape(tf.concat(tf.split(x, [1]*n_channels, axis=-1), 0),
                                                 tuple([n_labels*n_channels])))(std_devs)
        std_devs_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([std_devs, cat_labels])

    # build images based on mean and std maps
    image = KL.multiply([std_devs_map, image])
    image = KL.add([image, means_map])

    return image
