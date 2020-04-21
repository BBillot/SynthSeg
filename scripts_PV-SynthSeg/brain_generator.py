# python imports
import os
import time
import logging
import numpy as np

# project imports
from SynthSeg.brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils

logging.getLogger('tensorflow').disabled = True

# ----------------------------------------------------- example 1 ------------------------------------------------------

# general parameters
n_examples = 3
result_folder = '../generated_images/'

# generation parameters
paths = './atlases'
generation_labels = './labels_classes_stats/generation_labels.npy'
segmentation_labels = './labels_classes_stats/segmentation_labels.npy'
batchsize = 1
channels = 2
target_resolution = 0.6  # in mm
output_shape = 96  # crop produced image to this size
output_divisible_by_n = None  # 16  # output image should have dimension divisible by n (e.g. for deep learning use)
flip = False
prior_means = './labels_classes_stats/means_range.npy'
prior_stds = './labels_classes_stats/std_devs_range.npy'
specific_stats_for_channel = True
generation_classes = '/data/PVSeg/labels_classes_stats/generation_classes.npy'
scaling_bounds = './labels_classes_stats/scaling_range.npy'
rotation_bounds = './labels_classes_stats/rotation_range.npy'
shearing_bounds = 0.015
nonlin_std = 4
background_blur = False
data_res = './labels_classes_stats/blurring_resolution.npy'
thickness = None
downsample = True
blur_range = 1.2
crop_channel_2 = './labels_classes_stats/cropping_stats_t2.npy'
bias_field_std_dev = 0.5

# ---------------------------------------------------- example 2 -------------------------------------------------------

# # general parameters
# n_examples = 10
# result_folder = './generated_images/b40_test'
#
# # generation parameters
# paths = './atlases_full_brain'
# generation_labels = './labels_classes_stats/generation_labels.npy'
# segmentation_labels = './labels_classes_stats/segmentation_labels.npy'
# batchsize = 1
# channels = 1
# target_resolution = None  # in mm
# output_shape = 160  # crop produced image to this size
# output_divisible_by_n = 32  # output image should have dimension divisible by n (e.g. for deep learning use)
# flip = True
# prior_means = './labels_classes_stats/means_range.npy'
# prior_stds = './labels_classes_stats/std_devs_range.npy'
# specific_stats_for_channel = True
# generation_classes = './labels_classes_stats/generation_classes.npy'
# scaling_bounds = None
# rotation_bounds = None
# shearing_bounds = None
# nonlin_std = 3
# background_blur = True
# data_res = './labels_classes_stats/blurring_resolution_6_1_1.npy'
# thickness = './labels_classes_stats/thickness_4_1_1.npy'
# downsample = True
# blur_range = 1.5
# crop_channel_2 = None
# bias_field_std_dev = 0.6


########################################################################################################

# load label list, classes list and intensity ranges if necessary
generation_labels, n_neutral_labels = utils.get_list_labels(generation_labels, FS_sort=True)
if segmentation_labels is not None:
    segmentation_labels, _ = utils.get_list_labels(segmentation_labels, FS_sort=True)
else:
    segmentation_labels = generation_labels

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=paths,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 batch_size=batchsize,
                                 n_channels=channels,
                                 target_res=target_resolution,
                                 output_shape=output_shape,
                                 output_div_by_n=output_divisible_by_n,
                                 flipping=flip,
                                 prior_distributions='normal',
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 use_specific_stats_for_channel=specific_stats_for_channel,
                                 generation_classes=generation_classes,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 nonlin_std_dev=nonlin_std,
                                 blur_background=background_blur,
                                 data_res=data_res,
                                 thickness=thickness,
                                 downsample=downsample,
                                 blur_range=blur_range,
                                 crop_channel_2=crop_channel_2,
                                 bias_field_std_dev=bias_field_std_dev)

if not os.path.exists(os.path.join(result_folder)):
    os.mkdir(result_folder)

for n in range(n_examples):

    # generate new image and corresponding labels
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('deformation {0:d} took {1:.01f}s'.format(n, end - start))

    # save image
    for b in range(batchsize):
        utils.save_volume(np.squeeze(im[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'minibatch_{}_image_{}.nii.gz'.format(n, b)))
        utils.save_volume(np.squeeze(lab[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'minibatch_{}_labels_{}.nii.gz'.format(n, b)))
