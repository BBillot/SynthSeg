import tensorflow as tf
from . import TestData
import nibabel as nib
import numpy as np
import SynthSeg.brain_generator as bg


def test_brain_generator():
    # tf.config.run_functions_eagerly(True)
    tf.random.set_seed(1234)
    np.random.seed(1234)
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0])
    im, lab = brain_generator.generate_brain()
    if TestData.debug_nifti_output:
        img_data = tf.squeeze(im)
        nib.save(nib.Nifti1Image(img_data.numpy(), np.eye(4)), TestData.get_tmp_output_dir() / "generated_brain.nii")
    assert im.shape == lab.shape, "Shape of the label image and the generated MRI are not the same"



