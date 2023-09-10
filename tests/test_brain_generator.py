import tensorflow as tf
from . import TestData
import nibabel as nib
import numpy as np
import SynthSeg.brain_generator as bg
import timeit
import pytest


def test_brain_generator(fixed_random_seed):
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0])
    im, lab = brain_generator.generate_brain()
    if TestData.debug_nifti_output:
        img_data = tf.squeeze(im)
        nib.save(
            nib.Nifti1Image(img_data.numpy(), np.eye(4)),
            TestData.get_test_output_dir() / "generated_brain.nii",
        )
    assert (
        im.shape == lab.shape
    ), "Shape of the label image and the generated MRI are not the same"


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


def test_tfrecords_compression(tmp_path):
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tf.keras.utils.set_random_seed(43)
    tfrecord = tmp_path / "test.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)
    size1 = tfrecord.stat().st_size

    tfrecord = tmp_path / "test2.tfrecord"
    brain_generator.generate_tfrecord(tfrecord, compression_type="GZIP")
    size2 = tfrecord.stat().st_size

    assert size1 / size2 > 40


@pytest.mark.parametrize("compression", ("", "GZIP"))
def test_read_tfrecords(tmp_path, compression):
    def measure_iteration(ds):
        def func():
            for _ in ds:
                pass

        return func

    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=3)

    tf.keras.utils.set_random_seed(43)
    for i in range(2):
        tfrecord = tmp_path / f"test{i}.tfrecord"
        brain_generator.generate_tfrecord(tfrecord, compression_type=compression)
    files = list(tmp_path.glob("*.tfrecord"))

    dataset = bg.read_tfrecords(files, compression_type=compression)
    time1 = timeit.timeit(measure_iteration(dataset), number=10)

    dataset = bg.read_tfrecords(
        files,
        num_parallel_reads=2,
        compression_type=compression,
    )
    time2 = timeit.timeit(measure_iteration(dataset), number=10)

    assert time1 > time2
