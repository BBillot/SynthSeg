"""

Very simple script to show how we trained SynthSeg+, which extends SynthSeg by building robustness to clinical
acquisitions. For more details, please look at our MICCAI 2022 paper:

Robust Segmentation of Brain MRI in the Wild with Hierarchical CNNs and no Retraining,
Billot, Magdamo, Das, Arnold, Iglesias
MICCAI 2022

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
from SynthSeg.training import training as training_s1
from SynthSeg.training_denoiser import training as training_d
from SynthSeg.training_group import training as training_s2

import numpy as np

# ------------------ segmenter S1
# Here the purpose is to train a first network to produce preliminary segmentations of input scans with five general
# labels: 0-background, 1-white matter, 2-grey matter, 3-fluids, 4-cerebellum.

# As in tutorial 3, S1 is trained with synthetic images with randomised contrasts/resolution/artefacts such that it can
# readily segment a wide range of test scans without retraining. The synthetic scans are obtained from the same label
# maps and generative model as in the previous tutorials.
labels_dir_s1 = '../../data/training_label_maps'
path_generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
path_generation_classes = '../../data/labels_classes_priors/generation_classes.npy'
# However, because we now wish to segment scans using only five labels, we use a different list of segmentation labels
# where all label values in generation_labels are assigned to a target value between [0, 4].
path_segmentation_labels_s1 = '../../data/tutorial_7/segmentation_labels_s1.npy'

model_dir_s1 = './outputs_tutorial_7/training_s1'  # folder where the models will be saved


training_s1(labels_dir=labels_dir_s1,
            model_dir=model_dir_s1,
            generation_labels=path_generation_labels,
            segmentation_labels=path_segmentation_labels_s1,
            n_neutral_labels=18,
            generation_classes=path_generation_classes,
            target_res=1,
            output_shape=160,
            prior_distributions='uniform',
            prior_means=[0, 255],
            prior_stds=[0, 50],
            randomise_res=True)

# ------------------ denoiser D
# The purpose of this network is to perform label-to-label correction in order to correct potential mistakes made by S1
# at test time. Therefore, D is trained with two sets of label maps: noisy segmentations from S1 (used as inputs to D),
# and their corresponding ground truth (used as target to train D). In order to obtain input segmentations
# representative of the mistakes of S1, these are obtained by degrading real images with extreme augmentation (spatial,
# intensity, resolution, etc.), and feeding them to S1.

# Obtaining the input/target segmentations is done offline by using the following function: sample_segmentation_pairs.py
# In practice we sample a lot of them (i.e. 10,000), but we give here 8 example pairs. Note that these segmentations
# have the same label values as the output of S1 (i.e. between [0, 4]).
list_input_labels = ['../../data/tutorial_7/noisy_segmentations_d/0001.nii.gz',
                     '../../data/tutorial_7/noisy_segmentations_d/0002.nii.gz',
                     '../../data/tutorial_7/noisy_segmentations_d/0003.nii.gz']
list_target_labels = ['../../data/tutorial_7/target_segmentations_d/0001.nii.gz',
                      '../../data/tutorial_7/target_segmentations_d/0002.nii.gz',
                      '../../data/tutorial_7/target_segmentations_d/0003.nii.gz']

# Moreover, we perform spatial augmentation on the sampled pairs, in order to further increase the morphological
# variability seen by the network. Furthermore, the input "noisy" segmentations are further augmented with random
# erosion/dilation:
prob_erosion_dilation = 0.3  # probability of performing random erosion/dilation
min_erosion_dilation = 4,    # minimum coefficient for erosion/dilation
max_erosion_dilation = 5     # maximum coefficient for erosion/dilation

# This is the list of label values included in the input/GT label maps. This list must contain unique values.
input_segmentation_labels = np.array([0, 1, 2, 3, 4])

model_dir_d = './outputs_tutorial_7/training_d'  # folder where the models will be saved

training_d(list_paths_input_labels=list_input_labels,
           list_paths_target_labels=list_target_labels,
           model_dir=model_dir_d,
           input_segmentation_labels=input_segmentation_labels,
           output_shape=160,
           prob_erosion_dilation=prob_erosion_dilation,
           min_erosion_dilation=min_erosion_dilation,
           max_erosion_dilation=max_erosion_dilation,
           conv_size=5,
           unet_feat_count=16,
           skip_n_concatenations=2)

# ------------------ segmenter S2
# Final segmentations are obtained with a last segmenter S2, which takes as inputs an image as well as the preliminary
# segmentations of S1 that are corrected by D.

# Here S2 is trained with synthetic images sampled from the usual training label maps with associated generation labels,
# classes. Also, we now use the same segmentation labels as in tutorials 2, 3, and 4, as we now segment all the usual
# regions.
labels_dir_s2 = '../../data/training_label_maps'  # these are the same as for S1
path_generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
path_generation_classes = '../../data/labels_classes_priors/generation_classes.npy'
path_segmentation_labels_s2 = '../../data/labels_classes_priors/synthseg_segmentation_labels.npy'

# The preliminary segmentations are given as soft probability maps and are directly derived from the ground truth.
# Specifically, we take the structures that were segmented by S1, and regroup them into the same "groups" as before.
grouping_labels = '../../data/tutorial_7/segmentation_labels_s1.npy'
# However, in order to simulate test-time imperfections made by D, we these soft probability maps are slightly
# augmented with spatial transforms, and sometimes undergo a random dilation/erosion.

model_dir_s2 = './outputs_tutorial_7/training_s2'  # folder where the models will be saved

training_s2(labels_dir=labels_dir_s2,
            model_dir=model_dir_s2,
            generation_labels=path_generation_labels,
            n_neutral_labels=18,
            segmentation_labels=path_segmentation_labels_s2,
            generation_classes=path_generation_classes,
            grouping_labels=grouping_labels,
            target_res=1,
            output_shape=160,
            prior_distributions='uniform',
            prior_means=[0, 255],
            prior_stds=[0, 50],
            randomise_res=True)
