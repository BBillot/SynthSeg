# This script shows typical parameters used for image generation in SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally

# python imports
import os
import time
import numpy as np

# project imports
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator


# Pool of label maps to generate images from.
# Each new image is generated from a label map randomly selected among the provided label maps.
path_label_maps = '../../data/training_label_maps'

# general parameters
n_examples = 2
result_folder = '../../generated_images'
output_shape = 160  # randomly crop produced image to this size
output_divisible_by_n = 32  # make sure the shape of the output images is divisible by 32 (overwrites output_shape)
flipping = True  # enable right/left flipping.

# here we use uniform prior distribution to sample the means/stds of the GMM. Because we don't specify prior_means and
# prior_stds, those priors will have default bounds of [25, 225], and [5, 25]. Those values enable to generate a wide
# range of contrasts (often unrealistic), which will make the segmentation network contrast-agnostic.
prior_distributions = 'uniform'

# set path to generation labels (labels to generate intensities from)
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# set path to the set of labels that we want to keep in the output label maps (called here segmentation labels)
segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# Because we enabled right/left flipping, and because our label map contains different labels for contralateral
# structures we need to sort the generation_labels between non-sided, left and right structures.
# Thus we directly load the generation labels here, and sort them according to FreeSurfer classification.
generation_labels, n_neutral_labels = utils.get_list_labels(generation_labels, FS_sort=True)

########################################################################################################

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_label_maps,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 output_shape=output_shape,
                                 output_div_by_n=output_divisible_by_n,
                                 prior_distributions=prior_distributions,
                                 flipping=flipping)

utils.mkdir(result_folder)

for n in range(n_examples):

    # generate new image and corresponding labels
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('deformation {0:d} took {1:.01f}s'.format(n, end - start))

    # save image
    utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'SynthSeg_image_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'SynthSeg_labels_%s.nii.gz' % n))
