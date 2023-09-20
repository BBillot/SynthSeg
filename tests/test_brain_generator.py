import tensorflow as tf
from . import TestData
import nibabel as nib
import numpy as np
import SynthSeg.brain_generator as bg
import timeit


def test_brain_generator():
    tf.keras.utils.set_random_seed(12345)
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

    tf.keras.utils.set_random_seed(43)
    tfrecord2 = tmp_path / "test2.tfrecord"
    brain_generator.generate_tfrecord(tfrecord2, compression_type="GZIP")

    assert tfrecord.stat().st_size / tfrecord2.stat().st_size > 2


def test_read_tfrecords(tmp_path):
    def measure_iteration(ds):
        def func():
            for _ in ds:
                pass

        return func

    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tf.keras.utils.set_random_seed(43)
    for i in range(10):
        tfrecord = tmp_path / f"test{i}.tfrecord"
        brain_generator.generate_tfrecord(tfrecord, compression_type="GZIP")
    files = list(tmp_path.glob("*.tfrecord"))

    dataset = bg.read_tfrecords(files, compression_type="GZIP")
    time1 = timeit.timeit(measure_iteration(dataset), number=10)

    dataset = bg.read_tfrecords(
        files,
        num_parallel_reads=2,
        compression_type="GZIP",
    )
    time2 = timeit.timeit(measure_iteration(dataset), number=10)

    print(time1, time2)
    assert time1 > time2
