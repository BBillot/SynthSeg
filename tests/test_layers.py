import tensorflow as tf
from tensorflow.python.keras.utils.tf_utils import ListWrapper
from ext.lab2im.edit_tensors import gaussian_kernel


def test_gaussian_blur2():
    """
    I'm not sure if this test-case has the correct input arguments for gaussian_kernel() but this is
    at least what I see in the debugger. The real shit-show happens in ext.lab2im.utils.reformat_to_list()
    which seems to be the cow that is milked all over the place.
    """
    sigma = tf.convert_to_tensor([[0.5, 0.5, 1.01]])
    max_sigma = ListWrapper([6.0, 6.0, 6.0])
    blur_range = 1.03
    separable = True
    kernels = gaussian_kernel(sigma, max_sigma, blur_range, separable)
    assert tf.is_tensor(kernels)


