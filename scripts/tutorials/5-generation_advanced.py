"""

This script shows how to generate synthetic images with narrowed intensity distributions (e.g. T1-weighted scans) and
at a specific resolution. All the arguments shown here can be used in the training function.
These parameters were not explained in the previous tutorials as they were not used for the training of SynthSeg.

Specifically, this script generates 5 examples of training data simulating 3mm axial T1 scans, which have been resampled
at 1mm resolution to be segmented.
Contrast-specificity is achieved by now imposing Gaussian priors (instead of uniform) over the GMM parameters.
Resolution-specificity is achieved by first blurring and downsampling to the simulated LR. The data will then be
upsampled back to HR, so that the downstream network is trained to segment at HR. This upsampling step mimics the
process that will happen at test time.



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
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# script parameters
n_examples = 5  # number of examples to generate in this script
result_dir = './outputs_tutorial_5'  # folder where examples will be saved


# path training label maps
path_label_map = '../../data/training_label_maps'
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
output_labels = '../../data/labels_classes_priors/synthseg_segmentation_labels.npy'
n_neutral_labels = 18
output_shape = 160


# ---------- GMM sampling parameters ----------

# Here we use Gaussian priors to control the means and standard deviations of the GMM.
prior_distributions = 'normal'

# Here we still regroup labels into classes of similar tissue types:
# Example: (continuing the example of tutorial 1)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                                 generation_classes = [0,  1,   2, 3, 4, 5,  4,  6,  3,  4,  5,  4,  6]
# Note that structures with right/left labels are now associated with the same class.
generation_classes = '../../data/labels_classes_priors/generation_classes_contrast_specific.npy'

# We specify here the hyperparameters governing the prior distribution of the GMM.
# As these prior distributions are Gaussian, they are each controlled by a mean and a standard deviation.
# Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the total number of classes specified
# in generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
# correspond to standard deviations.
#
# Example: (continuing the previous one) prior_means = np.array([[0, 30, 80, 110, 95, 40, 70]
#                                                                [0, 10, 50,  15, 10, 15, 30]])
# This means that intensities of label 3 and 17, which are both in class 4, will be drawn from the Gaussian
# distribution, whose mean will be sampled from the Gaussian distribution with index 4 in prior_means N(95, 10).
# Here is the complete table of correspondence for this example:
# mean of Gaussian for label   0 drawn from N(0,0)=0
# mean of Gaussian for label  24 drawn from N(30,10)
# mean of Gaussian for label 507 drawn from N(80,50)
# mean of Gaussian for labels 2 and 41 drawn from N(110,15)
# mean of Gaussian for labels 3, 17, 42, 53 drawn from N(95,10)
# mean of Gaussian for labels 4 and 43 drawn from N(40,15)
# mean of Gaussian for labels 25 and 57 drawn from N(70,30)
# These hyperparameters were estimated with the function SynthSR/estimate_priors.py/build_intensity_stats()
prior_means = '../../data/labels_classes_priors/prior_means_t1.npy'
# same as for prior_means, but for the standard deviations of the GMM.
prior_stds = '../../data/labels_classes_priors/prior_stds_t1.npy'

# ---------- Resolution parameters ----------

# here we aim to synthesise data at a specific resolution, thus we do not randomise it anymore !
randomise_res = False

# blurring/downsampling parameters
# We specify here the slice spacing/thickness that we want the synthetic scans to mimic. The axes refer to the *RAS*
# axes, as all the provided data (label maps and images) will be automatically aligned to those axes during training.
# RAS refers to Right-left/Anterior-posterior/Superior-inferior axes, i.e. sagittal/coronal/axial directions.
data_res = np.array([1., 1., 3.])  # slice spacing i.e. resolution to mimic
thickness = np.array([1., 1., 3.])  # slice thickness

# ------------------------------------------------------ Generate ------------------------------------------------------

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_label_map,
                                 generation_labels=generation_labels,
                                 output_labels=output_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 output_shape=output_shape,
                                 prior_distributions=prior_distributions,
                                 generation_classes=generation_classes,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 randomise_res=randomise_res,
                                 data_res=data_res,
                                 thickness=thickness)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'image_t1_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'labels_t1_%s.nii.gz' % n))
