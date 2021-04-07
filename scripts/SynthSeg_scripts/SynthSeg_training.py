# This script shows typical parameters used to train SynthSeg (segmentation of 1mm MRI scans of any contrast):
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally

# We strongly recommend that you generate a couple of training examples before launching any training script. This will
# allow you to check that everything looks normal, with the generation parameters that you have selected.


# project imports
from SynthSeg.training import training


# path training label maps
path_training_label_maps = '../../data/training_label_maps'
# path of directory where the models will be saved during training
path_model_dir = '../../models/SynthSeg_training'

# set path to generation labels (i.e. the structure to represent in the synthesised scans)
path_generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
path_segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# generation parameters
target_res = None  # resolution of the output segmentation (the resolution of the training label maps by default)
output_shape = 160  # size of the generated examples (obtained by random cropping), tune this to the size of your GPU
batchsize = 1  # number of training examples per mini-batch, tune this to the size of your GPU
n_channels = 1  # here we work in the uni-modal case, so we synthetise only one channel

# spatial deformation paramaters
flipping = True  # whether to right/left flip the training label maps, this will take sided labels into account
# (so that left labels are indeed on the left when the label map is flipped)
scaling_bounds = .15  # the following are for linear spatial deformation, higher is more deformation
rotation_bounds = 15
shearing_bounds = .012
translation_bounds = False  # here we deactivate translation as we randomly crop the training examples
nonlin_std = 3.  # maximum strength of the elastic deformation, higher enables more deformation
nonlin_shape_factor = .04  # scale at which to elastically deform, higher is more local deformation

# bias field parameters
bias_field_std = .5  # maximum strength of the bias field, higher enables more corruption
bias_shape_factor = .025  # scale at which to sample the bias field, lower is more constant across the image

# acquisition resolution parameters
# the following parameters aim at mimicking data that would have been 1) acquired at low resolution (i.e. data_res),
# and 2) upsampled to high resolution in order to obtain segmentation at high res (see target_res).
# We do not such effects here, as this script shows training parameters to segment data at 1mm isotropic resolution
data_res = None
randomise_res = False
thickness = None
downsample = False
blur_range = 1.03  # we activate this parameter, which enables SynthSeg to be robust against small resolution variations

# architecture parameters
n_levels = 5  # number of resolution levels
nb_conv_per_level = 2  # number of convolution per level
conv_size = 3  # size of the convolution kernel (e.g. 3x3x3)
unet_feat_count = 24  # number of feature maps after the first convolution
# if feat_multiplier is set to 1, we will keep the number of feature maps constant throughout the network; 2 will double
# them(resp. half) after each max-pooling (resp. upsampling); 3 will triple them, etc.
feat_multiplier = 2
dropout = 0
activation = 'elu'  # activation for all convolution layers except the last, which will use sofmax regardless

# training parameters
lr = 1e-4  # learning rate
lr_decay = 0
wl2_epochs = 1  # number of pre-training epochs
dice_epochs = 100  # number of training epochs
steps_per_epoch = 5000

training(labels_dir=path_training_label_maps,
         model_dir=path_model_dir,
         path_generation_labels=path_generation_labels,
         path_segmentation_labels=path_segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         flipping=flipping,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         nonlin_shape_factor=nonlin_shape_factor,
         data_res=data_res,
         thickness=thickness,
         downsample=downsample,
         blur_range=blur_range,
         bias_field_std=bias_field_std,
         bias_shape_factor=bias_shape_factor,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch)
