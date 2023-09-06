import tensorflow as tf

from SynthSeg.training_with_tfrecords import training
from SynthSeg.training_options import TrainingOptions
from SynthSeg.brain_generator import BrainGenerator

from . import TestData


def test_training(tmp_path):
    tf.keras.utils.set_random_seed(43)

    label_map_files = TestData.get_label_maps()
    brain_generator = BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tfrecord = tmp_path / "train.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)

    training(
        TrainingOptions(
            data_dir=str(tmp_path),
            model_dir=str(tmp_path / "output"),
            wl2_epochs=1,
            dice_epochs=1,
            steps_per_epoch=2,
            batchsize=1,
        )
    )

    output_files = [p.name for p in (tmp_path / "output").iterdir()]
    assert all([f in output_files for f in ["wl2_001.h5", "dice_001.h5"]])
