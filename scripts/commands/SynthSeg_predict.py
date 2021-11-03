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


"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
print('\n')
print('SynthSeg prediction')
print('\n')

# python imports
import os
import sys
from argparse import ArgumentParser

# add main folder to python path and import ./SynthSeg/predict.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
from SynthSeg.predict import predict


# parse arguments
parser = ArgumentParser()

# input/outputs
parser.add_argument("--i", type=str, dest='path_images',
                    help="Image(s) to segment. Can be a path to an image or to a folder.")
parser.add_argument("--o", type=str, dest="path_segmentations",
                    help="Segmentation output(s). Must be a folder if --i designates a folder.")
parser.add_argument("--post", type=str, default=None, dest="path_posteriors",
                    help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
parser.add_argument("--resample", type=str, default=None, dest="path_resampled",
                    help="(optional) Resampled image(s). Must be a folder if --i designates a folder.")
parser.add_argument("--vol", type=str, default=None, dest="path_volumes",
                    help="(optional) Output CSV file with volumes for all structures and subjects.")

# parameters
parser.add_argument("--crop", nargs='+', type=int, default=192, dest="cropping",
                    help="(optional) Size of 3D patches to analyse. Default is 192.")
parser.add_argument("--threads", type=int, default=1, dest="threads",
                    help="(optional) Number of cores to be used. Default is 1.")
parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")

# parse commandline
args = vars(parser.parse_args())

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
del args['cpu']

# limit the number of threads to be used if running on CPU
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(args['threads'])
del args['threads']

# default parameters
args['segmentation_labels'] = os.path.join(synthseg_home, 'data/labels_classes_priors/segmentation_labels.npy')
args['n_neutral_labels'] = 18
args['segmentation_label_names'] = os.path.join(synthseg_home, 'data/labels_classes_priors/segmentation_names.npy')
args['topology_classes'] = os.path.join(synthseg_home, 'data/labels_classes_priors/topological_classes.npy')
args['path_model'] = os.path.join(synthseg_home, 'models/SynthSeg.h5')
args['padding'] = args['cropping']

# call predict
predict(**args)
