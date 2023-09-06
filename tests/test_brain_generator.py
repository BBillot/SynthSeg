import tensorflow as tf
from . import TestData
import nibabel as nib
import numpy as np
import SynthSeg.brain_generator as bg


def test_brain_generator(fixed_random_seed):
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0])
    im, lab = brain_generator.generate_brain()
    if TestData.debug_nifti_output:
        img_data = tf.squeeze(im)
        nib.save(nib.Nifti1Image(img_data.numpy(), np.eye(4)), TestData.get_test_output_dir() / "generated_brain.nii")
    assert im.shape == lab.shape, "Shape of the label image and the generated MRI are not the same"

def test_tfrecords(tmp_path):
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tf.keras.utils.set_random_seed(43)
    image, labels = brain_generator.generate_brain()

    tf.keras.utils.set_random_seed(43)
    tfrecord = tmp_path / "test.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)
    image2, labels2 = brain_generator.tfrecord_to_brain(tfrecord)

    np.testing.assert_array_equal(image, image2)
    np.testing.assert_array_equal(labels, labels2)
