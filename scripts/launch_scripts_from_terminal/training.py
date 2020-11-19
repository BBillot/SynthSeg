from argparse import ArgumentParser
from SynthSeg.training import training

parser = ArgumentParser()

# Positional arguments
parser.add_argument("labels_dir", type=str)
parser.add_argument("model_dir", type=str)

# ---------------------------------------------- Generation parameters ----------------------------------------------
# label maps parameters
parser.add_argument("--generation_labels", type=str, dest="path_generation_labels", default=None)
parser.add_argument("--segmentation_labels", type=str, dest="path_segmentation_labels", default=None)
parser.add_argument("--save_generation_labels", type=str, dest="save_generation_labels", default=None)

# output-related parameters
parser.add_argument("--batch_size", type=int, dest="batchsize", default=1)
parser.add_argument("--channels", type=int, dest="n_channels", default=1)
parser.add_argument("--target_res", type=float, dest="target_res", default=None)
parser.add_argument("--output_shape", type=int, dest="output_shape", default=None)

# GMM-sampling parameters
parser.add_argument("--generation_classes", type=str, dest="path_generation_classes", default=None)
parser.add_argument("--prior_type", type=str, dest="prior_distributions", default='uniform')
parser.add_argument("--prior_means", type=str, dest="prior_means", default=None)
parser.add_argument("--prior_stds", type=str, dest="prior_stds", default=None)
parser.add_argument("--specific_stats", action='store_true', dest="use_specific_stats_for_channel")
parser.add_argument("--mix_prior_and_random", action='store_true', dest="mix_prior_and_random")

# spatial deformation parameters
parser.add_argument("--no_flipping", action='store_false', dest="flipping")
parser.add_argument("--scaling", dest="scaling_bounds", default=None)
parser.add_argument("--rotation", dest="rotation_bounds", default=None)
parser.add_argument("--shearing", dest="shearing_bounds", default=None)
parser.add_argument("--nonlin_std", type=float, dest="nonlin_std", default=3.)
parser.add_argument("--nonlin_shape_factor", type=float, dest="nonlin_shape_factor", default=.04)

# blurring/resampling parameters
parser.add_argument("--no_background_blurring", action='store_false', dest="blur_background")
parser.add_argument("--data_res", type=str, dest="data_res", default=None)
parser.add_argument("--thickness", type=str, dest="thickness", default=None)
parser.add_argument("--downsample", action='store_true', dest="downsample")
parser.add_argument("--blur_range", type=float, dest="blur_range", default=1.15)
parser.add_argument("--crop_channel_2", type=str, dest="crop_channel_2", default=None)

# bias field parameters
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=.3)
parser.add_argument("--bias_shape_factor", type=float, dest="bias_shape_factor", default=.025)

# ------------------------------------------ UNet architecture parameters ------------------------------------------
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5)
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2)
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3)
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24)
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2)
parser.add_argument("--dropout", type=float, dest="dropout", default=0.)
parser.add_argument("--no_batch_norm", action='store_true', dest="no_batch_norm")
parser.add_argument("--activation", type=str, dest="activation", default='elu')

# ----------------------------------------------- Training parameters ----------------------------------------------
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", default=0)
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs", default=5)
parser.add_argument("--dice_epochs", type=int, dest="dice_epochs", default=100)
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000)
parser.add_argument("--background_weight", type=float, dest="background_weight", default=1e-4)
parser.add_argument("--include_background", action='store_true', dest="include_background")
parser.add_argument("--loss_cropping", type=int, dest="loss_cropping", default=None)
parser.add_argument("--load_model_file", type=str, dest="load_model_file", default=None)
parser.add_argument("--initial_epoch_wl2", type=int, dest="initial_epoch_wl2", default=0)
parser.add_argument("--initial_epoch_dice", type=int, dest="initial_epoch_dice", default=0)

args = parser.parse_args()
training(**vars(args))
