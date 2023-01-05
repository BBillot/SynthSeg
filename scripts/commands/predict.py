"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# imports
from argparse import ArgumentParser
from SynthSeg.predict import predict

parser = ArgumentParser()

# Positional arguments
parser.add_argument("path_images", type=str, help="path single image or path of the folders with training labels")
parser.add_argument("path_segmentations", type=str, help="segmentations folder/path")
parser.add_argument("path_model", type=str, help="model file path")

# labels parameters
parser.add_argument("labels_segmentation", type=str, help="path label list")
parser.add_argument("--neutral_labels", type=int, dest="n_neutral_labels", default=None)
parser.add_argument("--names_list", type=str, dest="names_segmentation", default=None,
                    help="path list of label names, only used if --vol is specified")

# Saving paths
parser.add_argument("--post", type=str, dest="path_posteriors", default=None, help="posteriors folder/path")
parser.add_argument("--resampled", type=str, dest="path_resampled", default=None,
                    help="path/folder of the images resampled at the given target resolution")
parser.add_argument("--vol", type=str, dest="path_volumes", default=None, help="path volume file")

# Processing parameters
parser.add_argument("--min_pad", type=int, dest="min_pad", default=None,
                    help="margin of the padding")
parser.add_argument("--cropping", type=int, dest="cropping", default=None,
                    help="crop volume before processing. Segmentations will have the same size as input image.")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="Target resolution at which segmentations will be given.")
parser.add_argument("--flip", action='store_true', dest="flip",
                    help="to activate test-time augmentation (right/left flipping)")
parser.add_argument("--topology_classes", type=str, dest="topology_classes", default=None,
                    help="path list of classes, for topologically enhanced biggest connected component analysis")
parser.add_argument("--smoothing", type=float, dest="sigma_smoothing", default=0.5,
                    help="var for gaussian blurring of the posteriors")
parser.add_argument("--biggest_component", action='store_true', dest="keep_biggest_component",
                    help="only keep biggest component in segmentation (recommended)")

# Architecture parameters
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3, help="size of unet convolution masks")
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels for unet")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2, help="conv par level")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24,
                    help="number of features of unet first layer")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="factor of new feature maps per level")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function")

# Evaluation parameters
parser.add_argument("--gt", type=str, default=None, dest="gt_folder",
                    help="folder containing ground truth segmentations, which triggers the evaluation.")
parser.add_argument("--eval_label_list", type=str, dest="evaluation_labels", default=None,
                    help="labels to evaluate Dice scores on if gt is provided. Default is the same as label_list.")
parser.add_argument("--incorrect_labels", type=str, default=None, dest="list_incorrect_labels",
                    help="path list labels to correct.")
parser.add_argument("--correct_labels", type=str, default=None, dest="list_correct_labels",
                    help="path list correct labels.")

args = parser.parse_args()
predict(**vars(args))
