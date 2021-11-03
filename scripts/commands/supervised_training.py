"""
If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# imports
from argparse import ArgumentParser
from SynthSeg.supervised_training import supervised_training
from ext.lab2im.utils import infer

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------
# Positional arguments
parser.add_argument("image_dir", type=str)
parser.add_argument("labels_dir", type=str)
parser.add_argument("model_dir", type=str)

parser.add_argument("--segmentation_labels", type=str, dest="segmentation_labels", default=None)
parser.add_argument("--neutral_labels", type=int, dest="n_neutral_labels", default=None)
parser.add_argument("--batch_size", type=int, dest="batchsize", default=1)
parser.add_argument("--target_res", type=int, dest="target_res", default=None)
parser.add_argument("--output_shape", type=int, dest="output_shape", default=None)

# ----------------------------------------------- Augmentation parameters ----------------------------------------------
# spatial deformation parameters
parser.add_argument("--no_flipping", action='store_false', dest="flipping")
parser.add_argument("--scaling", dest="scaling_bounds", type=infer, default=.15)
parser.add_argument("--rotation", dest="rotation_bounds", type=infer, default=15)
parser.add_argument("--shearing", dest="shearing_bounds", type=infer, default=.012)
parser.add_argument("--translation", dest="translation_bounds", type=infer, default=False)
parser.add_argument("--nonlin_std", type=float, dest="nonlin_std", default=3.)
parser.add_argument("--nonlin_shape_factor", type=float, dest="nonlin_shape_factor", default=.04)

# ressampling parameters
parser.add_argument("--data_res", dest="data_res", type=infer, default=None)
parser.add_argument("--thickness", dest="thickness", type=infer, default=None)
parser.add_argument("--downsample", action='store_true', dest="downsample")
parser.add_argument("--blur_range", type=float, dest="blur_range", default=1.03)

# bias field parameters
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=.5)
parser.add_argument("--bias_shape_factor", type=float, dest="bias_shape_factor", default=.025)

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5)
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2)
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3)
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24)
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2)
parser.add_argument("--activation", type=str, dest="activation", default='elu')

# ------------------------------------------------- Training parameters ------------------------------------------------
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", default=0)
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs", default=5)
parser.add_argument("--dice_epochs", type=int, dest="dice_epochs", default=100)
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000)
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None)

args = parser.parse_args()
supervised_training(**vars(args))
