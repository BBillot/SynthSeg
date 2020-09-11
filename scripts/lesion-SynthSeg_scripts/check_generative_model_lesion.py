from SynthSeg.check_generative_model import check_generative_model

# new outputs
tensor_names = [
    'labels_lesions',
    'image_out',
    'labels_out'
]
filenames = [
    'labels_lesions_save',
    'image_out_save',
    'labels_out_save'
]

# ------------------------------------------------- Le's data ----------------------------------------------------------

# # general parameters
# examples = 50
# result_folder = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/generated_images'

# # generation parameters
# labels_folder = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/training_labels'
# vae_model = '/home/benjamin/PycharmProjects/SynthSeg/VAE/decoder_keras_model.h5'
# generation_labels = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/generation_labels.npy'
# segmentation_labels = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/segmentation_labels.npy'
# batchsize = 1
# channels = 1
# target_resolution = 1
# output_shape = None
# output_divisible_by_n = None
# generation_classes = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/generation_classes.npy'
# prior_distribution = 'normal'
# prior_means = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/prior_means_from_ms.npy'
# prior_stds = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/prior_stds_from_ms.npy'
# specific_stats_for_channel = True
# flip = False
# apply_linear_trans = True
# scaling_bounds = None
# rotation_bounds = None
# shearing_bounds = None
# apply_nonlin_trans = True
# nonlin_std = 2
# background_blur = True
# data_res = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/blurring_res.npy'
# thickness = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/thickness.npy'
# downsample = True
# blur_range = 1.2
# crop_channel_2 = None
# apply_bias_field = True
# bias_field_std = 0.3

# -------------------------------------------------- MS data -----------------------------------------------------------

# general parameters
examples = 3
result_folder = '/home/benjamin/data/lesions/MS/PV-SynthSeg/generated_images'

# generation parameters
labels_folder = '/home/benjamin/data/Buckner40/labels/training/extra_cerebral_generation_rl_regrouped'
vae_model = '/home/benjamin/PycharmProjects/SynthSeg/VAE/decoder_keras_model.h5'
generation_labels = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/generation_labels.npy'
segmentation_labels = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/segmentation_labels.npy'
batchsize = 1
channels = 2
target_resolution = 1
output_shape = None
output_divisible_by_n = None
generation_classes = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/generation_classes.npy'
prior_distribution = 'normal'
prior_means = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/prior_means_t1_flair.npy'
prior_stds = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/prior_stds_t1_flair.npy'
specific_stats_for_channel = True
flip = False
apply_linear_trans = True
scaling_bounds = None
rotation_bounds = None
shearing_bounds = None
apply_nonlin_trans = True
nonlin_std = 2
background_blur = True
data_res = None
thickness = None
downsample = False
blur_range = 1.2
crop_channel_2 = None
apply_bias_field = True
bias_field_std = 0.3

check_generative_model(labels_folder,
                       vae_model,
                       examples,
                       tensor_names,
                       filenames,
                       result_folder,
                       generation_labels=generation_labels,
                       segmentation_labels=segmentation_labels,
                       batchsize=batchsize,
                       n_channels=channels,
                       target_res=target_resolution,
                       output_shape=output_shape,
                       output_div_by_n=output_divisible_by_n,
                       generation_classes=generation_classes,
                       prior_distributions=prior_distribution,
                       prior_means=prior_means,
                       prior_stds=prior_stds,
                       use_specific_stats_for_channel=specific_stats_for_channel,
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