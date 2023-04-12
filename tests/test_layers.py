import tensorflow as tf
from tensorflow.python.keras.utils.tf_utils import ListWrapper

import nibabel as nib
import numpy as np
import pathlib

from . import TestData

from ext.lab2im.edit_tensors import gaussian_kernel
from ext.lab2im.layers import GaussianBlur


def test_gaussian_blur():
    layer = GaussianBlur(sigma=5.0)
    x_in = tf.pad(tf.ones((1, 1, 1)), paddings=tf.constant([[10, 10], [10, 10], [10, 10]]))
    x_in = tf.reshape(x_in, [1] + list(x_in.shape) + [1])
    y_out = layer(x_in)
    if TestData.debug_nifti_output:
        img_data = tf.squeeze(y_out)
        nib.save(nib.Nifti1Image(img_data.numpy(), np.eye(4)), TestData.get_tmp_output_dir() / "blurred_cube.nii")
    assert x_in.shape == y_out.shape


def test_gaussian_blur_kernel():
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


