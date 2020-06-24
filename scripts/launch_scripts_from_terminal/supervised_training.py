# imports
from argparse import ArgumentParser
from SynthSeg.supervised_training import training


parser = ArgumentParser()

# Positional arguments
parser.add_argument("image_dir", type=str, help="path of the folders with training images")
parser.add_argument("labels_dir", type=str, help="path of the folders with training labels")

# Generation parameters
parser.add_argument("--crop", type=int, default=None, dest="cropping", help="cropping size")
parser.add_argument("--no_flipping", action='store_false', dest="flipping", help="prevent right/left flipping")
parser.add_argument("--scaling", default=0.15, dest="scaling_range", help="scaling range")
parser.add_argument("--rotation", default=15, dest="rotation_range", help="rotation range")
parser.add_argument("--shearing", default=0.01, dest="shearing_range", help="shearing range")
parser.add_argument("--nonlin_std", type=float, default=3, dest="nonlin_std_dev",
                    help="variance of SVF before up-sampling")
parser.add_argument("--nonlin_shape_fact", type=float, default=0.04, dest="nonlin_shape_fact",
                    help="size of the initial SVF for the non-linear deformation.")
parser.add_argument("--crop_channel_2", type=str, default=None, dest="crop_channel_2", help="cropping for channel2")

# Architecture parameters
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3, help="size of unet's convolution masks")
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels for unet")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2, help="conv par level")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="ratio of new feature maps per level")
parser.add_argument("--dropout", type=float, dest="dropout", default=0, help="dropout probability")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24,
                    help="number of features in first layer of the Unet")
parser.add_argument("--no_batch_norm", action='store_true', dest="no_batch_norm",
                    help="deactivate batch normalisation")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function")

# Training parameters
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--lr_decay", type=float, dest="lr_decay", default=0, help="learning rate decay")
parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="batch_size")
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs", default=2, help="number of iterations")
parser.add_argument("--dice_epochs", type=int, dest="dice_epochs", default=20, help="number of iterations")
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=2500,
                    help="frequency of model saves")
parser.add_argument("--background_weight", type=float, dest="background_weight", default=0.0001,
                    help="background of weighted l2 training")
parser.add_argument("--include_background", action='store_true', dest="include_background",
                    help="whether to include the background label in the dice metric")

# Resuming parameters
parser.add_argument("--load_model_file", type=str, dest="load_model_file", default=None,
                    help="optional h5 model file to initialize with. Will be loaded to wl2 model if wl2_epochs>0,"
                         "and to dice model otherwise")
parser.add_argument("--initial_epoch_wl2", type=int, dest="initial_epoch_wl2", default=0,
                    help="initial epoch for wl2 pretraining model, useful when resuming wl2 training")
parser.add_argument("--initial_epoch_dice", type=int, dest="initial_epoch_dice", default=0,
                    help="initial epoch for dice model, useful when resuming dice model training")

# Load materials
parser.add_argument("--load_label_list", type=str, dest="path_label_list", default=None,
                    help="load already computed label list for image generation. "
                         "Default is None: the label list is computed during the script execution.")

# Saving paths
parser.add_argument("--model_dir", type=str, dest="model_dir", default='./models/ben_models/',
                    help="models folder")

args = parser.parse_args()
training(**vars(args))