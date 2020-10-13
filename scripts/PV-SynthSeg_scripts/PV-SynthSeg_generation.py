# This script shows typical parameters used for image generation in PV-SynthSeg:
# Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast
# Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias

# In particular we show here advanced options to create multi-channel scans mimicking anisotropically acquired data.

# python imports
import os
import time
import numpy as np

# project imports
from SynthSeg.brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils


# common parameters to all examples
n_examples = 3
result_folder = '../../generated_images'
path_label_maps = '../../data/training_label_maps'

# set path to generation labels (labels to generate intensities from)
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# set path to the set of labels that we want to keep in the output label maps (called here segmentation labels)
segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# GMM-sampling parameters
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'
prior_distribution = 'normal'
prior_means = '../../data/labels_classes_priors/prior_means.npy'  # the same prior will be used for all channels
prior_stds = '../../data/labels_classes_priors/prior_stds.npy'

# blurring parameter
# Here we keep want to obtain images with realistic background, either full-zero or low-intenisty Gaussian.
# This options also enable correction for edge bluring effects.
background_blur = False

# IMPORTANT !!!
# Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
# these values refer to the RAS axes.

# ---------------------------------------------------- multi_modal -----------------------------------------------------
# Here we want to mimick multi-modal scans (double T1w, this is just an example) acquired at 1mm and subsampled to 1.5mm

# output-related parameters
channels = 2  # create multi-modal image with 2 channels
target_resolution = 1.5  # resample output to 1.5mm isotropic (input label map is at 1mm isotropic)
output_shape = None  # we don't impose a specific size for output...
output_divisible_by_n = 16  # ... but we want it to be divisible by 16 (e.g. for deep learning use)

# spatial deformation parameters
scaling_bounds = np.array([[0.8, 0.95, 0.95], [1.2, 1.05, 1.05]])  # exagerated scaling in first dimension
rotation_bounds = 15  # the rotation angles will be drawn from the same uniform distribution [-15, 15] for all axes
shearing_bounds = [-.01, .012]  # same uniform distribution for all axes, slighlty biased towards positive values
nonlin_std = 4  # increase the effect of the elastic deformation (default value is 3)

# blurring/resampling parameters
# since we already sample the synthetic scans at 1mm isotropic, we don't need to downsample them to *acquisition* res
data_res = None  # same as input label maps
thickness = None  # slice thickness is the same as spacing
downsample = False  # don't downsample at acquisition resolution
blur_range = 1.2  # introduce some randomness in blurring to make the network adaptive to small resolution variations

# bias field parameters
bias_field_std = 0.4  # we increase the strength of the applied bias field (default is 0.3)

# -------------------------------------------------- anisotropic T1w ---------------------------------------------------
# Here we want to mimick scans acquired at 6x1x1mm resolution and upsampled at 1mm isotropic, with slice thickness
# of 4mm in the first axis (sagittal).
# uncomment to run this example (don't forget to comment out the other example)

# # output-related parameters
# channels = 1
# target_resolution = None  # same resolution as input label maps, so 1mm isotropic
# output_shape = 160  # randomly crop produced image to this size
# output_divisible_by_n = 32  # make sure output shape is divisible by 32
#
# # spatial deformation parameters
# scaling_bounds = None  # keep default value of [.85, 1.15]
# rotation_bounds = None  # keep default value of [-15, 15]
# shearing_bounds = None  # keep default value of [-.01, .01]
# nonlin_std = 3  # keep default value
#
# # blurring/resampling parameters
# data_res = np.array([6, 1, 1])  # resolution of the data we want to mimick
# thickness = np.array([4, 1, 1])  # slice thickess of the data we want tot mimick
# # as the input label map is at 1mm isotropic, so will be the images sampled from the GMM. Thus, in
# # order to make the generated images realistic, we want to downsample them to acquisiton resolution
# downsample = True
# blur_range = 1.5  # introduce some randomness in blurring to make the network adaptive to small resolution variations
#
# # bias field parameters
# bias_field_std = .3  # keep default value

# -------------------------------------- multi-modal with different resolutions ----------------------------------------
# Here we want to mimick multi-modal scans with a first channel acquired at 6x1x1mm, and a seond channel at 1x9x1, both
# with a slice thickness of 4mm in the acquisition direction.
# uncomment to run this example (don't forget to comment out the first example)

# # output-related parameters
# channels = 2
# target_resolution = None  # same resolution as input label maps, so 1mm isotropic
# output_shape = None
# output_divisible_by_n = None
#
# # spatial deformation parameters (keep default values)
# scaling_bounds = None
# rotation_bounds = None
# shearing_bounds = None
# nonlin_std = 3
#
# # blurring/resampling parameters
# data_res = np.array([[6, 1, 1], [1, 9, 1]])  # resolution of the data we want to mimick
# thickness = np.array([[4, 1, 1], [1, 4, 1]])  # slice thickess of the data we want tot mimick
# downsample = True
# blur_range = 1.5
#
# # bias field parameters (keep default value)
# bias_field_std = .3


########################################################################################################

# load label list, classes list and intensity ranges if necessary
generation_labels, n_neutral_labels = utils.get_list_labels(generation_labels, FS_sort=True)

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_label_maps,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 n_channels=channels,
                                 target_res=target_resolution,
                                 output_shape=output_shape,
                                 output_div_by_n=output_divisible_by_n,
                                 generation_classes=generation_classes,
                                 prior_distributions=prior_distribution,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 nonlin_std=nonlin_std,
                                 blur_background=background_blur,
                                 data_res=data_res,
                                 thickness=thickness,
                                 downsample=downsample,
                                 blur_range=blur_range,
                                 bias_field_std=bias_field_std)

utils.mkdir(result_folder)

for n in range(n_examples):

    # generate new image and corresponding labels
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('deformation {0:d} took {1:.01f}s'.format(n, end - start))

    # save image
    utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'PV-SynthSeg_image_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'PV-SynthSeg_labels_%s.nii.gz' % n))
