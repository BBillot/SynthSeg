from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Union

import simple_parsing

from SynthSeg.training import training


def int_or_str(arg: str):
    try:
        return int(arg)
    except ValueError:
        return arg


# TODO: copy doc strings from train function to argument help, maybe separate into different data classes ...
@dataclass
class Options:
    path_training_label_maps: str = (
        "../../data/training_label_maps"  # Label maps for the training
    )
    path_model_dir: str = "./output_tutorials_3"  # Output path

    # architecture parameters

    n_levels: int = 5  # Number of resolution levels
    nb_conv_per_level: int = 2  # Number of convolution per level
    conv_size: int = 3  # Size of the convolution kernel (e.g. 3x3x3)
    unet_feat_count: int = 24  # Number of feature maps after the first convolution
    activation: str = "elu"  # Activation for all convolution layers except the last, which will use softmax regardless
    feat_multiplier: int = 2  # If this is set to 1, we will keep the number of feature maps constant throughout the network; 2 will double them (resp. half) after each max-pooling (resp. upsampling); 3 will triple them, etc.

    # training parameters

    batchsize: int = 1  # Batch size
    lr: float = 1e-4  # learning rate
    wl2_epochs: int = 1  # Number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax
    dice_epochs: int = 100  # Number of training epochs
    steps_per_epoch: int = 5000  # Number of iterations per epoch

    # generation and segmentation labels

    path_generation_labels: str = "../../data/labels_classes_priors/generation_labels.npy"  # Path to generation labels
    n_neutral_labels: int = 18  # Number of neutral labels
    path_segmentation_labels: str = (
        "../../data/labels_classes_priors/synthseg_segmentation_labels.npy"
    )

    # shape and resolution of the output

    target_res: Optional[int] = None  # Target resolution
    output_shape: int = 160  # Output shape
    n_channels: int = 1  # Number of channels

    # GMM sampling

    prior_distributions: str = "uniform"  # The prior distribution
    path_generation_classes: str = "../../data/labels_classes_priors/generation_classes.npy"  # Path to the generation classes

    # spatial deformation parameters

    flipping: bool = True  # Flip?
    scaling_bounds: Union[float, bool] = 0.2  # Scaling bounds
    rotation_bounds: int = 15  # Rotation bounds
    shearing_bounds: float = 0.012  # Shearing bounds
    translation_bounds: Union[float, bool] = False  # Translation bounds
    nonlin_std: float = 4.0  # Nonlin std.
    bias_field_std: float = 0.7  # Bias field std.

    # acquisition resolution parameters

    randomise_res: bool = True  # Don't randomise resolution
    checkpoint: Optional[str] = None  # Path of an already saved model to load before starting the training.

    # WandB

    wandb_log_freq: Union[int, str] = "epoch"  # Log frequency for the WandB callback. If 'epoch', logs metrics at the end of each epoch. If 'batch', logs metrics at the end of each batch. If an integer, logs metrics at the end of that many batches. Defaults to 'epoch'."
    wandb: bool = False  # Log training to WandB


if __name__ == "__main__":
    args: Options = simple_parsing.parse(Options)

    training(
        args.path_training_label_maps,
        args.path_model_dir,
        generation_labels=args.path_generation_labels,
        segmentation_labels=args.path_segmentation_labels,
        n_neutral_labels=args.n_neutral_labels,
        batchsize=args.batchsize,
        n_channels=args.n_channels,
        target_res=args.target_res,
        output_shape=args.output_shape,
        prior_distributions=args.prior_distributions,
        generation_classes=args.path_generation_classes,
        flipping=not args.flipping,
        scaling_bounds=args.scaling_bounds,
        rotation_bounds=args.rotation_bounds,
        shearing_bounds=args.shearing_bounds,
        translation_bounds=args.translation_bounds,
        nonlin_std=args.nonlin_std,
        randomise_res=not args.randomise_res,
        bias_field_std=args.bias_field_std,
        n_levels=args.n_levels,
        nb_conv_per_level=args.nb_conv_per_level,
        conv_size=args.conv_size,
        unet_feat_count=args.unet_feat_count,
        feat_multiplier=args.feat_multiplier,
        activation=args.activation,
        lr=args.lr,
        wl2_epochs=args.wl2_epochs,
        dice_epochs=args.dice_epochs,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint=args.checkpoint,
        wandb=args.wandb,
        wandb_log_freq=args.wandb_log_freq,
    )
