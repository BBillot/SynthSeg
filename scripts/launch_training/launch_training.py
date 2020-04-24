# imports
from argparse import ArgumentParser
from SynthSeg.training import training


parser = ArgumentParser()

# Positional arguments
parser.add_argument("labels_dir", type=str)
parser.add_argument("model_dir", type=str)

# Generation parameters
parser.add_argument("--generation_label_list", type=str, dest="path_generation_labels")
parser.add_argument("--segmentation_label_list", type=str, dest="path_segmentation_labels")
parser.add_argument("--save_generation_labels", type=str, dest="save_generation_labels")
parser.add_argument("--batch_size", type=int, dest="batch_size")
parser.add_argument("--channels", type=int, dest="n_channels")
parser.add_argument("--target_res", type=float, dest="target_res")
parser.add_argument("--output_shape", type=int, dest="output_shape")
parser.add_argument("--generation_classes", type=str, dest="path_generation_classes")
parser.add_argument("--prior_type", type=str, dest="prior_distributions")
parser.add_argument("--prior_means", type=str, dest="prior_means")
parser.add_argument("--prior_stds", type=str, dest="prior_stds")
parser.add_argument("--specific_stats", action='store_true', dest="use_specific_stats_for_channel")
parser.add_argument("--no_flipping", action='store_false', dest="flipping")
parser.add_argument("--scaling", dest="scaling_bounds")
parser.add_argument("--rotation", dest="rotation_bounds")
parser.add_argument("--shearing", dest="shearing_bounds")
parser.add_argument("--nonlin_std", dest="nonlin_std")
parser.add_argument("--nonlin_shape_fact", type=float, dest="nonlin_shape_factor")
parser.add_argument("--no_background_blurring", action='store_false', dest="blur_background")
parser.add_argument("--data_res", dest="data_res")
parser.add_argument("--thickness", dest="thickness")
parser.add_argument("--downsample", dest="downsample")
parser.add_argument("--blur_range", dest="blur_range")
parser.add_argument("--crop_channel_2", type=str, dest="crop_channel_2")
parser.add_argument("--bias_std", type=float, dest="bias_field_std")
parser.add_argument("--bias_shape_factor", type=float, dest="bias_shape_factor")

# Architecture parameters
parser.add_argument("--n_levels", type=int, dest="n_levels")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level")
parser.add_argument("--conv_size", type=int, dest="conv_size")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier")
parser.add_argument("--dropout", type=float, dest="dropout")
parser.add_argument("--no_batch_norm", action='store_true', dest="no_batch_norm")

# Training parameters
parser.add_argument("--lr", type=float, dest="lr")
parser.add_argument("--lr_decay", type=float, dest="lr_decay")
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs")
parser.add_argument("--dice_epochs", type=int, dest="dice_epochs")
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch")
parser.add_argument("--background_weight", type=float, dest="background_weight")
parser.add_argument("--include_background", action='store_true', dest="include_background")
parser.add_argument("--loss_cropping", type=int, dest="loss_cropping")

# Training Resuming parameters
parser.add_argument("--load_model_file", type=str, dest="load_model_file")
parser.add_argument("--initial_epoch_wl2", type=int, dest="initial_epoch_wl2")
parser.add_argument("--initial_epoch_dice", type=int, dest="initial_epoch_dice")

args = parser.parse_args()
training(**vars(args))