import os
from typing import Tuple
from contextlib import nullcontext
from dataclasses import dataclass
from inspect import getmembers, isclass
from pathlib import Path

import tensorflow as tf
from simple_parsing.helpers.serialization import Serializable

from ext.lab2im import layers
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models

from .metrics_model import WeightedL2Loss, DiceLoss, IdentityLoss
from .option_types import *


class NullStrategy:
    @staticmethod
    def scope():
        return nullcontext()


@dataclass
class TrainingOptions(Serializable):
    """Options for a training with an already generated dataset, e.i. we already have synthetic brain images.

    Args:
        data_dir: Path to the data dir.
        output_dir: Path to the output dir.
        exist_ok:
        checkpoint:
        lr: Learning rate for the training.
        batch_size:
        wl2_epochs: Number of epochs for which the network (except the soft-max layer) is trained with L2
            norm loss function.
        dice_epochs: Number of epochs with the soft Dice loss function.
        steps_per_epoch: Number of steps per epoch.
        n_levels:
        nb_conv_per_level:
        conv_size:
        unet_feat_count:
        feat_multiplier:
        activation:
        strategy: Define a distributed training strategy: 'mirrored' or 'multiworker'.
            If None, do not distribute the training.
        wandb:
        wandb_log_freq:
    """

    data_dir: Path
    output_dir: Path
    exist_ok: bool = False
    checkpoint: Optional[Path] = None
    lr: float = 1e-4
    batch_size: int = 1
    wl2_epochs: int = 1
    dice_epochs: int = 50
    steps_per_epoch: Optional[int] = None
    n_levels: int = 5
    nb_conv_per_level: int = 2
    conv_size: int = 3
    unet_feat_count: int = 24
    feat_multiplier: int = 2
    activation: str = "elu"
    strategy: Optional[str] = None
    wandb: bool = False
    wandb_log_freq: Union[str, int] = "epoch"


def training(opts: TrainingOptions):
    """Train the U-net with a TFRecord Dataset.

    Args:
        opts: The training options
    """
    # Check epochs
    assert (opts.wl2_epochs > 0) | (
        opts.dice_epochs > 0
    ), "either wl2_epochs or dice_epochs must be positive, had {0} and {1}".format(
        opts.wl2_epochs, opts.dice_epochs
    )

    # Define distributed strategy
    if opts.strategy is None:
        strategy = NullStrategy()
    elif opts.strategy == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    elif opts.strategy == "multiworker":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise NotImplementedError(f"The '{opts.strategy}' strategy is not implemented.")

    # Create output dir
    opts.output_dir.mkdir(parents=True, exist_ok=opts.exist_ok)

    # Create dataset from tfrecords
    dataset = read_tfrecords(opts.data_dir)

    # Get output shape and number of labels from first example of the dataset
    for example in dataset.take(1):
        input_shape, nb_labels = example[0][0].shape, example[1].shape[-1]

    # Batch dataset
    dataset = dataset.batch(opts.batch_size)

    checkpoint = opts.checkpoint

    # Define and compile model
    with strategy.scope():
        # prepare the segmentation model
        unet_model = nrn_models.unet(
            input_shape=input_shape,
            nb_labels=nb_labels,
            nb_levels=opts.n_levels,
            nb_conv_per_level=opts.nb_conv_per_level,
            conv_size=opts.conv_size,
            nb_features=opts.unet_feat_count,
            feat_mult=opts.feat_multiplier,
            activation=opts.activation,
            batch_norm=-1,
            name="unet",
        )

        # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
        if opts.wl2_epochs > 0:
            wl2_model = tf.keras.models.Model(
                unet_model.inputs, [unet_model.get_layer("unet_likelihood").output]
            )
            wl2_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=opts.lr),
                loss=WeightedL2Loss(n_labels=nb_labels),
            )

    if opts.wl2_epochs > 0:
        callbacks = build_callbacks(
            output_dir=opts.output_dir,
            metric_type="wl2",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
        )

        # fit
        wl2_model.fit(
            dataset,
            epochs=opts.wl2_epochs,
            steps_per_epoch=opts.steps_per_epoch,
            callbacks=callbacks,
        )

        if opts.wandb:
            import wandb

            wandb.finish()

        checkpoint = opts.output_dir / ("wl2_%03d.h5" % opts.wl2_epochs)

    if opts.dice_epochs > 0:
        with strategy.scope():
            # fine-tuning with dice metric
            dice_model, is_compiled, init_epoch = load_model(
                model=unet_model, checkpoint=checkpoint, metric_type="dice"
            )
            if not is_compiled:
                dice_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=opts.lr),
                    loss=DiceLoss(),
                )

        callbacks = build_callbacks(
            output_dir=opts.output_dir,
            metric_type="dice",
            wandb=opts.wandb,
            wandb_log_freq=opts.wandb_log_freq,
        )

        dice_model.fit(
            dataset,
            epochs=opts.dice_epochs,
            steps_per_epoch=opts.steps_per_epoch,
            callbacks=callbacks,
            initial_epoch=init_epoch,
        )


def read_tfrecords(data_dir: Path) -> tf.data.Dataset:
    def parse_example(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
        labels = tf.io.parse_tensor(example["labels"], out_type=tf.int32)

        return (image,), labels

    files = list(data_dir.glob("*.tfrecord"))
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_example)

    return dataset


def load_model(
    model: tf.keras.models.Model,
    checkpoint: Path,
    metric_type: str,
    reinitialise_momentum: bool = False,
) -> Tuple[tf.keras.models.Model, bool, int]:
    is_compiled = False
    init_epoch = 0

    if checkpoint is not None:
        if metric_type in checkpoint.name:
            init_epoch = int(checkpoint.name.split(metric_type)[1][1:-3])
        if (not reinitialise_momentum) & (metric_type in checkpoint.name):
            custom_l2i = {
                key: value
                for (key, value) in getmembers(layers, isclass)
                if key != "Layer"
            }
            custom_nrn = {
                key: value
                for (key, value) in getmembers(nrn_layers, isclass)
                if key != "Layer"
            }
            custom_objects = {
                **custom_l2i,
                **custom_nrn,
                "tf": tf,
                "keras": tf.keras,
                "loss": IdentityLoss().loss,
            }
            model = tf.keras.models.load_model(
                checkpoint, custom_objects=custom_objects
            )
            is_compiled = True
        else:
            model.load_weights(checkpoint, by_name=True)

    return model, is_compiled, init_epoch


def build_callbacks(
    output_dir: Path,
    metric_type,
    wandb: bool = False,
    wandb_log_freq: Union[str, int] = "epoch",
):
    # create log folder
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # model saving callback
    save_file_name = os.path.join(output_dir, "%s_{epoch:03d}.h5" % metric_type)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(save_file_name, verbose=1)]

    # TensorBoard callback
    if metric_type == "dice":
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False
            )
        )

    # WandB callback
    if wandb:
        import wandb as wandbm
        from wandb.integration.keras import WandbMetricsLogger

        wandbm.init()
        callbacks.append(WandbMetricsLogger(log_freq=wandb_log_freq))

    return callbacks
