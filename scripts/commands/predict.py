# imports
from argparse import ArgumentParser
from SynthSeg.predict import predict

parser = ArgumentParser()

# Positional arguments
parser.add_argument("path_images", type=str, help="path single image or path of the folders with training labels")

# Load materials
parser.add_argument("--model", type=str, default=None, dest="path_model", help="model file path")
parser.add_argument("--label_list", type=str, dest="segmentation_label_list", default=None,
                    help="path label list")
parser.add_argument("--names_list", type=str, dest="segmentation_names_list", default=None,
                    help="path list of label names")

# Saving paths
parser.add_argument("--out_seg", type=str, dest="path_segmentations", default=None, help="segmentations folder/path")
parser.add_argument("--out_post", type=str, dest="path_posteriors", default=None, help="posteriors folder/path")
parser.add_argument("--out_vol", type=str, dest="path_volumes", default=None, help="path volume file")
parser.add_argument("--out_resampled", type=str, dest="path_resampled", default=None,
                    help="path/folder of the resampled images (1mm isotropic resolution)")

# Processing parameters
parser.add_argument("--padding", type=int, dest="padding", default=None,
                    help="margin of the padding")
parser.add_argument("--cropping", type=int, dest="cropping", default=None,
                    help="crop volume before processing. Segmentations will have the same size as input image.")
parser.add_argument("--flip", action='store_true', dest="flip",
                    help="to activate test-time augmentation (right/left flipping)")
parser.add_argument("--topology_classes", type=str, dest="topology_classes", default=None,
                    help="path list of classes, for topologically enhanced biggest connected component analysis")
parser.add_argument("--smoothing", type=float, dest="sigma_smoothing", default=0,
                    help="var for gaussian blurring of the posteriors")
parser.add_argument("--biggest_component", action='store_true', dest="keep_biggest_component",
                    help="only keep biggest component in segmentation")

# Architecture parameters
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3, help="size of unet's convolution masks")
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels for unet")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2, help="conv par level")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24,
                    help="number of features of Unet's first layer")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="factor of new feature maps per level")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function")

# Evaluation parameters
parser.add_argument("--gt", type=str, default=None, dest="gt_folder",
                    help="folder containing ground truth segmentations, which triggers the evaluation.")
parser.add_argument("--mask", type=str, default=None, dest="mask_folder",
                    help="folder containing masks to mask out areas of the obtained segmentations.")
parser.add_argument("--incorrect_labels", type=str, default=None, dest="list_incorrect_labels",
                    help="path list labels to correct.")
parser.add_argument("--correct_labels", type=str, default=None, dest="list_correct_labels",
                    help="path list correct labels.")
parser.add_argument("--eval_label_list", type=str, dest="evaluation_label_list", default=None,
                    help="labels to evaluate Dice scores on if gt is provided. Default is the same as label_list.")

args = parser.parse_args()
predict(**vars(args))
