# This script shows typical parameters used to train SynthSeg:
# A Learning Strategy for Contrast-agnostic MRI Segmentation, MIDL 2020
# Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca*
# *contributed equally


# project imports
from SynthSeg.training import training


# path training label maps
path_training_label_maps = '../../data/training_label_maps'
# path of directory where to save the models during training
path_model_dir = '../../models/SynthSeg_training'

# set path to generation labels
generation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)
segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# generation parameters
target_res = 1  # resolution of the output segmentation
output_shape = 160  # tune this to the size of your GPU

# training parameters
wl2_epochs = 5
dice_epochs = 150
steps_per_epoch = 1000

training(labels_dir=path_training_label_maps,
         model_dir=path_model_dir,
         path_generation_labels=generation_labels,
         path_segmentation_labels=segmentation_labels,
         target_res=target_res,
         output_shape=output_shape,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch)
