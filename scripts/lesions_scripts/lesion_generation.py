# python imports
import os
import time
import numpy as np

# project imports
from SynthSeg.brain_generator import BrainGenerator

# third party imports
from ext.lab2im import utils


# general parameters
examples = 10
result_folder = '/home/benjamin/data/lesions/MS/SynthSeg/generated_images'

# generation parameters
labels_folder = '/home/benjamin/data/lesions/MS/labels/resample_1_1_1/samseg_lesions_extracerebral'
vae_model = None  # '/home/benjamin/PycharmProjects/SynthSeg/VAE/decoder_challenge.h5'
vae_mode = None  # 'challenge'
path_lesion_prior = None  # '/home/benjamin/data/lesions/MS/SynthSeg-VAE/registrations_buckner/lesion_prior_eye_padded.nii.gz'
path_lesion_maps = None  # '/home/benjamin/data/lesions/MS/SynthSeg-VAE/labels/challenge_aligned_to_buckner_template'
generation_labels = '/home/benjamin/data/lesions/MS/labels_classes_stats/generation_labels.npy'
segmentation_labels = '/home/benjamin/data/lesions/MS/labels_classes_stats/segmentation_labels.npy'
batchsize = 1
channels = 1
target_resolution = 1
output_shape = None
output_divisible_by_n = None
generation_classes = '/home/benjamin/data/lesions/MS/labels_classes_stats/generation_classes.npy'
prior_distributions = 'uniform'
prior_means = '/home/benjamin/data/lesions/MS/labels_classes_stats/prior_means_flair.npy'
prior_stds = '/home/benjamin/data/lesions/MS/labels_classes_stats/prior_stds_flair.npy'
specific_stats_for_channel = True
mix_prior_and_random = False
flip = True
apply_linear_trans = True
scaling_bounds = None
rotation_bounds = None
shearing_bounds = None
apply_nonlin_trans = True
nonlin_std = 2
background_blur = True
data_res = '/home/benjamin/data/lesions/MS/longitudinal_dataset/labels_classes_stats_longitudinal/longitudinal_flair_res_4.4.npy'
thickness = None
downsample = True
blur_range = 1.25
crop_channel_2 = None
apply_bias_field = True
bias_field_std = 0.3

########################################################################################################

# load label list, classes list and intensity ranges if necessary
generation_labels, n_neutral_labels = utils.get_list_labels(generation_labels, FS_sort=True)
if segmentation_labels is not None:
    segmentation_labels, _ = utils.get_list_labels(segmentation_labels, FS_sort=True)
else:
    segmentation_labels = generation_labels

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=labels_folder,
                                 vae_model=vae_model,
                                 vae_mode=vae_mode,
                                 path_lesion_prior=path_lesion_prior,
                                 path_lesion_maps=path_lesion_maps,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 batchsize=batchsize,
                                 n_channels=channels,
                                 target_res=target_resolution,
                                 output_shape=output_shape,
                                 output_div_by_n=output_divisible_by_n,
                                 generation_classes=generation_classes,
                                 prior_distributions=prior_distributions,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 use_specific_stats_for_channel=specific_stats_for_channel,
                                 mix_prior_and_random=mix_prior_and_random,
                                 flipping=flip,
                                 apply_linear_trans=apply_linear_trans,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 apply_nonlin_trans=apply_nonlin_trans,
                                 nonlin_std=nonlin_std,
                                 blur_background=background_blur,
                                 data_res=data_res,
                                 thickness=thickness,
                                 downsample=downsample,
                                 blur_range=blur_range,
                                 crop_channel_2=crop_channel_2,
                                 apply_bias_field=apply_bias_field,
                                 bias_field_std=bias_field_std)

utils.mkdir(result_folder)

for n in range(examples):

    # generate new image and corresponding labels
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('minibatch {0:d} took {1:.01f}s'.format(n, end - start))

    # save image
    for b in range(batchsize):
        utils.save_volume(np.squeeze(im[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'image_{0}_{1}.nii.gz'.format(n, b)))
        utils.save_volume(np.squeeze(lab[b, ...]), brain_generator.aff, brain_generator.header,
                          os.path.join(result_folder, 'labels_{0}_{1}.nii.gz'.format(n, b)))
