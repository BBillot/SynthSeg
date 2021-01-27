# This script shows typical parameters used to train PV-SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally


# imports
import numpy as np
from SynthSeg.training import training


# path training label maps
path_training_label_maps = '../../data/training_label_maps'
path_model_dir = '../../models/PV-SynthSeg_training'

# set path to generation labels and segmentation labels
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# prior distribution of the GMM
generation_classes = '../../data/labels_classes_priors/generation_classes.npy'
prior_means = '../../data/labels_classes_priors/prior_means.npy'
prior_stds = '../../data/labels_classes_priors/prior_stds.npy'

# generation parameters
target_res = 1
output_shape = 160  # tune this to the size of your GPU
data_res = np.array([6, 1, 1])  # acquisition resolution of the data we want to mimick
thickness = np.array([4, 1, 1])  # slice thickess of the data we want tot mimick
downsample = True  # downsample to acquisition resolution before resampling to target resolution
blur_range = 1.5

# training parameters
wl2_epochs = 5
dice_epochs = 100
steps_per_epoch = 1000

training(labels_dir=path_training_label_maps,
         model_dir=path_model_dir,
         path_generation_labels=generation_labels,
         path_segmentation_labels=segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         path_generation_classes=generation_classes,
         data_res=data_res,
         thickness=thickness,
         downsample=downsample,
         blur_range=blur_range,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch)
