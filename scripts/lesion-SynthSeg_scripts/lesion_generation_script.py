# python imports
import os
import time
import numpy as np

# project imports
from SynthSeg.brain_generator import BrainGenerator

# third party imports
from ext.lab2im import utils


# general parameters
n_examples = 10
result_folder = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/generated_images'

# generation parameters
labels_folder = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/training_labels'
vae_model = '/home/benjamin/PycharmProjects/SynthSeg/VAE/decoder_keras_model.h5'
path_generation_labels = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/generation_labels.npy'
path_segmentation_labels = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/segmentation_labels.npy'
batchsize = 1
channels = 1
target_resolution = None  # in mm
output_shape = None
output_divisible_by_n = None
prior_distributions = 'normal'
generation_classes = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/generation_classes.npy'
prior_means = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/prior_means_from_ms.npy'
prior_stds = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/prior_stds_from_ms.npy'
specific_stats_for_channel = False
flip = True
scaling = None
rotation = None
shearing = None
nonlin_std_dev = 4
background_blur = True
data_resolution = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/blurring_res.npy'
slice_thickness = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/thickness.npy'
peform_downsample = True
blurring_range = 1.2
crop_channel2 = None
bias_field_std_dev = 0.3

########################################################################################################

# load label list, classes list and intensity ranges if necessary
generation_labels, n_neutral_labels = utils.get_list_labels(path_generation_labels, FS_sort=True)
if path_segmentation_labels is not None:
    segmentation_labels, _ = utils.get_list_labels(path_segmentation_labels, FS_sort=True)
else:
    segmentation_labels = generation_labels

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=labels_folder,
                                 vae_model=vae_model,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 batchsize=batchsize,
                                 n_channels=channels,
                                 target_res=target_resolution,
                                 output_shape=output_shape,
                                 output_div_by_n=output_divisible_by_n,
                                 prior_distributions=prior_distributions,
                                 generation_classes=generation_classes,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 use_specific_stats_for_channel=specific_stats_for_channel,
                                 flipping=flip,
                                 scaling_bounds=scaling,
                                 rotation_bounds=rotation,
                                 shearing_bounds=shearing,
                                 nonlin_std=nonlin_std_dev,
                                 blur_background=background_blur,
                                 data_res=data_resolution,
                                 thickness=slice_thickness,
                                 downsample=peform_downsample,
                                 blur_range=blurring_range,
                                 crop_channel_2=crop_channel2,
                                 bias_field_std=bias_field_std_dev)

if not os.path.exists(os.path.join(result_folder)):
    os.mkdir(result_folder)

for n in range(n_examples):

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
