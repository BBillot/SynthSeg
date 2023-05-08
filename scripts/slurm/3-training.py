from argparse import ArgumentParser

from SynthSeg.training import training


if __name__ == "__main__":
    parser = ArgumentParser()

    # TODO: copy doc strings from train function to argument help
    # path training label maps
    parser.add_argument(
        "--path_training_label_maps",
        type=str,
        help="Output path",
        default="../../data/training_label_maps",
    )
    parser.add_argument(
        "--path_model_dir", type=str, help="Output path", default="./output_tutorials_3"
    )
    parser.add_argument("--batchsize", type=int, help="Batch size", default=1)

    # architecture parameters
    parser.add_argument(
        "--n_levels", type=int, help="Number of resolution levels", default=5
    )
    parser.add_argument(
        "--nb_conv_per_level",
        type=int,
        help="Number of convolution per level",
        default=2,
    )
    parser.add_argument(
        "--conv_size",
        type=int,
        help="Size of the convolution kernel (e.g. 3x3x3)",
        default=3,
    )
    parser.add_argument(
        "--unet_feat_count",
        type=int,
        help="Number of feature maps after the first convolution",
        default=24,
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="Activation for all convolution layers except the last, which will use softmax regardless",
        default="elu",
    )
    parser.add_argument(
        "--feat_multiplier",
        type=int,
        help="If this is set to 1, we will keep the number of feature maps constant throughout the network; "
        "2 will double them (resp. half) after each max-pooling (resp. upsampling); 3 will triple them, etc.",
        default=2,
    )

    # training parameters
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=1e-4,
    )
    parser.add_argument(
        "--wl2_epochs",
        type=int,
        help="Number of pre-training epochs with wl2 metric w.r.t. the layer before the softmax.",
        default=1,
    )
    parser.add_argument(
        "--dice_epochs",
        type=int,
        help="Number of training epochs.",
        default=100,
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        help="Number of iterations per epoch.",
        default=5000,
    )

    # generation and segmentation labels
    parser.add_argument(
        "--path_generation_labels",
        type=str,
        help="Path to generation labels.",
        default="../../data/labels_classes_priors/generation_labels.npy",
    )
    parser.add_argument(
        "--n_neutral_labels", type=int, help="Number of neutral labels.", default=18
    )
    parser.add_argument(
        "--path_segmentation_labels",
        type=str,
        help="Path to segmentation labels.",
        default="../../data/labels_classes_priors/synthseg_segmentation_labels.npy",
    )

    # shape and resolution of the outputs
    parser.add_argument(
        "--target_res", type=int, help="Target resolution.", default=None
    )
    parser.add_argument("--output_shape", type=int, help="Output shape.", default=160)
    parser.add_argument("--n_channels", type=int, help="Number of channels.", default=1)

    # GMM sampling
    parser.add_argument(
        "--prior_distributions",
        type=str,
        help="The prior distribution.",
        default="uniform",
    )
    parser.add_argument(
        "--path_generation_classes",
        type=str,
        help="Path to the generation classes.",
        default="../../data/labels_classes_priors/generation_classes.npy",
    )

    # spatial deformation parameters
    parser.add_argument("--no_flipping", help="Don't flip.", action="store_true")
    parser.add_argument(
        "--scaling_bounds",
        type=float,
        help="Scaling bounds.",
        default=0.2,
    )
    parser.add_argument(
        "--rotation_bounds",
        type=int,
        help="Rotation bounds.",
        default=15,
    )
    parser.add_argument(
        "--shearing_bounds",
        type=float,
        help="Shearing bounds.",
        default=0.012,
    )
    parser.add_argument(
        "--translation_bounds", help="Translation bounds.", action="store_true"
    )
    parser.add_argument(
        "--nonlin_std",
        type=float,
        help="Nonlin std.",
        default=4.0,
    )
    parser.add_argument(
        "--bias_field_std",
        type=float,
        help="Bias field std.",
        default=0.7,
    )

    # acquisition resolution parameters
    parser.add_argument(
        "--no_randomise_res", help="Don't randomise resolution.", action="store_true"
    )

    # wandb
    parser.add_argument(
        "--wandb", help="Log training to WandB.", action="store_true"
    )

    args = parser.parse_args()

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
        flipping=not args.no_flipping,
        scaling_bounds=args.scaling_bounds,
        rotation_bounds=args.rotation_bounds,
        shearing_bounds=args.shearing_bounds,
        translation_bounds=args.translation_bounds,
        nonlin_std=args.nonlin_std,
        randomise_res=not args.no_randomise_res,
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
        wandb=args.wandb,
    )
