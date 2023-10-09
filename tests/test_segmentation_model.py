import tensorflow as tf
import numpy as np
from SynthSeg.training_options import TrainingOptions
from SynthSeg.training_with_tfrecords import training


def test_unet_integration(tmp_path, tfrecord):
    tf.keras.utils.set_random_seed(43)

    opts = TrainingOptions(
        model_dir=str(tmp_path / "output"),
        wl2_epochs=5,
        dice_epochs=10,
        steps_per_epoch=None,
        batchsize=1,
        tfrecords_dir=str(tfrecord.path.parent),
        input_shape=list(tfrecord.shape),
        n_labels=tfrecord.n_labels,
        use_original_unet=True,
        n_levels=3,
        lr=0.0001,
    )

    old_losses = training(opts).history["loss"]

    tf.keras.utils.set_random_seed(43)

    opts = TrainingOptions(
        model_dir=str(tmp_path / "output"),
        wl2_epochs=5,
        dice_epochs=10,
        steps_per_epoch=None,
        batchsize=1,
        tfrecords_dir=str(tfrecord.path.parent),
        input_shape=list(tfrecord.shape),
        n_labels=tfrecord.n_labels,
        use_original_unet=False,
        n_levels=3,
        lr=0.0001,
    )

    new_losses = training(opts).history["loss"]

    np.testing.assert_allclose(new_losses, old_losses, rtol=0.01)
