"""

This script explains how the different parameters controlling the generation of the synthetic data.
These parameters will be reused in the training function, but we describe them here, as the synthetic images are saved,
and thus can be visualised.
Note that most of the parameters here are set to their default value, but we show them nonetheless, just to explain
their effect. Moreover, we encourage the user to play with them to get a sense of their impact on the generation.



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


import os
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# script parameters
n_examples = 5  # number of examples to generate in this script
result_dir = './outputs_tutorial_2'  # folder where examples will be saved


# ---------- Input label maps and associated values ----------

# folder containing label maps to generate images from (note that they must have a ".nii", ".nii.gz" or ".mgz" format)
path_label_map = '../../data/training_label_maps'

# Here we specify the structures in the label maps for which we want to generate intensities.
# This is given as a list of label values, which do not necessarily need to be present in every label map.
# However, these labels must follow a specific order: first the background, and then all the other labels. Moreover, if
# 1) the label maps contain some right/left-specific label values, and 2) we activate flipping augmentation (which is
# true by default), then the rest of the labels must follow a strict order:
# first the non-sided labels (i.e. those which are not right/left specific), then all the left labels, and finally the
# corresponding right labels (in the same order as the left ones). Please make sure each that each sided label has a
# right and a left value (this is essential!!!).
#
# Example: generation_labels = [0,    # background
#                               24,   # CSF
#                               507,  # extra-cerebral soft tissues
#                               2,    # left white matter
#                               3,    # left cerebral cortex
#                               4,    # left lateral ventricle
#                               17,   # left hippocampus
#                               25,   # left lesions
#                               41,   # right white matter
#                               42,   # right cerebral cortex
#                               43,   # right lateral ventricle
#                               53,   # right hippocampus
#                               57]   # right lesions
# Note that plenty of structures are not represented here..... but it's just an example ! :)
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'


# We also have to specify the number of non-sided labels in order to differentiate them from the labels with
# right/left values.
# Example: (continuing the previous one): in this example it would be 3 (background, CSF, extra-cerebral soft tissues).
n_neutral_labels = 18

# By default, the output label maps (i.e. the target segmentations) contain all the labels used for generation.
# However, we may want not to predict all the generation labels (e.g. extra-cerebral soft tissues).
# For this reason, we specify here the target segmentation label corresponding to every generation structure.
# This new list must have the same length as generation_labels, and follow the same order.
#
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
# Note that in this example the labels 24 (CSF), and 507 (extra-cerebral soft tissues) are not predicted, or said
# differently they are segmented as background.
# Also, the left and right lesions (labels 25 and 57) are segmented as left and right white matter (labels 2 and 41).
output_labels = '../../data/labels_classes_priors/synthseg_segmentation_labels.npy'


# ---------- Shape and resolution of the outputs ----------

# number of channel to synthesise for multi-modality settings. Set this to 1 (default) in the uni-modality scenario.
n_channels = 1

# We have the possibility to generate training examples at a different resolution than the training label maps (e.g.
# when using ultra HR training label maps). Here we want to generate at the same resolution as the training label maps,
# so we set this to None.
target_res = None

# The generative model offers the possibility to randomly crop the training examples to a given size.
# Here we crop them to 160^3, such that the produced images fit on the GPU during training.
output_shape = 160


# ---------- GMM sampling parameters ----------

# Here we use uniform prior distribution to sample the means/stds of the GMM. Because we don't specify prior_means and
# prior_stds, those priors will have default bounds of [0, 250], and [0, 35]. Those values enable to generate a wide
# range of contrasts (often unrealistic), which will make the segmentation network contrast-agnostic.
prior_distributions = 'uniform'

# We regroup labels with similar tissue types into K "classes", so that intensities of similar regions are sampled
# from the same Gaussian distribution. This is achieved by providing a list indicating the class of each label.
# It should have the same length as generation_labels, and follow the same order. Importantly the class values must be
# between 0 and K-1, where K is the total number of different classes.
#
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                        generation_classes = [0,  1,   2, 3, 4, 5,  4,  6,  7,  8,  9,  8, 10]
# In this example labels 3 and 17 are in the same *class* 4 (that has nothing to do with *label* 4), and thus will be
# associated to the same Gaussian distribution when sampling the GMM.
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'


# ---------- Spatial augmentation ----------

# We now introduce some parameters concerning the spatial deformation. They enable to set the range of the uniform
# distribution from which the corresponding parameters are selected.
# We note that because the label maps will be resampled with nearest neighbour interpolation, they can look less smooth
# than the original segmentations.

flipping = True  # enable right/left flipping
scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
nonlin_std = 4.  # this controls the maximum elastic deformation (higher = more deformation)
bias_field_std = 0.7  # this controls the maximum bias field corruption (higher = more bias)


# ---------- Resolution parameters ----------

# This enables us to randomise the resolution of the produces images.
# Although being only one parameter, this is crucial !!
randomise_res = True


# ------------------------------------------------------ Generate ------------------------------------------------------

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_label_map,
                                 generation_labels=generation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 prior_distributions=prior_distributions,
                                 generation_classes=generation_classes,
                                 output_labels=output_labels,
                                 n_channels=n_channels,
                                 target_res=target_res,
                                 output_shape=output_shape,
                                 flipping=flipping,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 translation_bounds=translation_bounds,
                                 nonlin_std=nonlin_std,
                                 bias_field_std=bias_field_std,
                                 randomise_res=randomise_res)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'image_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'labels_%s.nii.gz' % n))
