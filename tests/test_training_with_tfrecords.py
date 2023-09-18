import tensorflow as tf
import numpy as np
import pytest

from SynthSeg.training_options import TrainingOptions
from SynthSeg.training_with_tfrecords import training
from SynthSeg.brain_generator import BrainGenerator

from . import TestData


@pytest.mark.parametrize(
    "wl2_epochs, dice_epochs, mean, std, exact, files",
    [
        (
            1,
            0,
            24.849784088134765,
            0.3270473375185669,
            24.895092010498047,
            ["wl2_001.h5"],
        ),
        (
            0,
            1,
            0.9801624119281769,
            0.001492286024485122,
            0.9799067378044128,
            ["dice_001.h5"],
        ),
        (
            1,
            1,
            0.979167926311493,
            0.0021950396925116346,
            0.9810947775840759,
            ["wl2_001.h5", "dice_001.h5"],
        ),
    ],
    ids=["wl2", "dice", "dice_after_wl2"],
)
def test_training(
    tmp_path, wl2_epochs, dice_epochs, mean, std, exact, files
):
    """
    Tests the equivalence with the original training via `mean` and `std`
    and the current implementation via `exact`.
    """
    tf.keras.utils.set_random_seed(43)

    label_map_files = TestData.get_label_maps()

    brain_generator = BrainGenerator(label_map_files[0], target_res=8, batchsize=2)

    tfrecord = tmp_path / "train.tfrecord"
    brain_generator.generate_tfrecord(tfrecord)

    opts = TrainingOptions(
        labels_dir=label_map_files[0],  # only needed for the experiment below
        target_res=8,  # only needed for the experiment below
        model_dir=str(tmp_path / "output"),
        wl2_epochs=wl2_epochs,
        dice_epochs=dice_epochs,
        steps_per_epoch=2,
        batchsize=1,
        tfrecords_dir=str(tmp_path),
        input_shape=32,
        n_labels=53,
    )

    results = training(opts)
    output_files = [p.name for p in (tmp_path / "output").iterdir()]

    # mean and std were obtained via the experiment below
    np.testing.assert_allclose(results.history["loss"][0], mean, atol=2*std)
    np.testing.assert_allclose(results.history["loss"][0], exact)
    assert all([f in output_files for f in files])

    # Experiment:
    # from SynthSeg.training import training_from_options
    # losses = np.empty(10)
    # for i in range(10):
    #     results = training_from_options(opts)
    #     losses[i] = results.history["loss"][0]
    # print(losses, losses.mean(), losses.std())
