"""

This script shows how to perform inference after having trained your own model.
Importantly, it reuses some of the parameters used in tutorial 3-training.
Moreover, we emphasise that this tutorial explains how to perform inference on your own trained models.
To predict segmentations based on the distributed mode for SynthSeg, please refer to the README.md file.



If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

# project imports
from SynthSeg.predict import predict

# paths to input/output files
# Here we assume the availability of an image that we wish to segment with a model we have just trained.
# We emphasise that we do not provide such an image (this is just an example after all :))
# Input images must have a .nii, .nii.gz, or .mgz extension.
# Note that path_images can also be the path to an entire folder, in which case all the images within this folder will
# be segmented. In this case, please provide path_segm (and possibly path_posteriors, and path_resampled) as folder.
path_images = '../../data/inputs/im.nii.gz'
# path to the output segmentation
path_segm = '../../data/outputs/im_seg.nii.gz'
# we can also provide paths for optional files containing the probability map for all predicted labels
path_posteriors = '../../data/outputs/im_post.nii.gz'
# and for a csv file that will contain the volumes of each segmented structure
path_vol = '../../data/outputs/volumes.csv'

# of course we need to provide the path to the trained model
path_model = '../../models/SynthSeg_training/dice_100.h5'
# but we also need to provide the path to the segmentation labels used during training
path_segmentation_labels = '../../data/labels_classes_priors/segmentation_labels.npy'

# We can now provide various parameters to control the preprocessing of the input.
# First we can play witht he size of the input. Remember that the size of input must be divisible by 2**n_levels, so the
# input image will be automatically cropped to such a shape. If you want to make sure that the whole image is processed,
# (and not to have ugly cropping artefacts in your segmentations) you can pad it to a larger size with zeros.
padding = 192  # pad the image to this shape if it is smaller
# On the contrary, if the ROI is much smaller than the actual image, you can crop the input to a smaller shape for
# faster processing, or to make it fit on your GPU.
cropping = 160
# Finally, we finish preprocessing the input by resampling it to the resolution at which the network has been trained to
# produce predictions. If the input image has a resolution outside the range [target_res-0.05, target_res+0.05], it will
# automatically be resampled to target_res. Note that if the image is indeed resampled, you have the option to save the
# resampled image.
target_res = 1.
path_resampled = '../../data/outputs/im_resampled_target_res.nii.gz'

# After the image has been processed by the network, there are again various options to postprocess it.
# First, we can apply some test-time augmentation by flipping the input along the right-left axis and segmenting
# the resulting image. In this case, and if the network has right/left specific labels, it is also very important to
# provide the number of neutral labels. This must be the exact same as the one used during training.
flip = True
n_neutral_labels = 18
# Second, we can smooth the probability maps produced by the network. This doesn't change much the results, but helps to
# reduce high frequency noise in the obtained segmentations.
sigma_smoothing = 0.5
# Then we can operate some fancier version of biggest connected component, by regrouping structures within so-called
# "topological classes". For each class we successively: 1) sum all the posteriors corresponding to the labels of this
# class, 2) obtain a mask for this class by thresholding the summed posteriors by a low value (arbitrarily set to 0.1),
# 3) keep the biggest connected component, and 4) individually apply the obtained mask to the posteriors of all the
# labels for this class.
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
#                                       topological_classes = [0,  0,  0,  1, 1, 2,  3,  1,  4,  4,  5,  6,  7]
# Here we regroup labels 2 and 3 in the same topological class, same for labels 41 and 42. The topological class of
# unsegmented structures must be set to 0 (like for 24 and 507).
topology_classes = '../../data/labels_classes_priors/topological_classes.npy'
# Finally, we can also operate a strict version of biggest connected component, to get rid of unwanted noisy label
# patch that can sometimes occur in the background. If so, we do recommend to use the smoothing option described above.
keep_biggest_component = True

# Regarding the architecture of the network, we must provide the predict function with the same parameters as during
# training.
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2

# Finally, we can set up an evaluation step after all images have been segmented.
# In this case, Dice scores will be computed and saved as a numpy array in the folder containing the segmentation(s).
# This numpy array will be organised as follows: rows correspond to structures, and columns to subjects. Importantly,
# rows are given in a sorted order.
# Example: we segment 2 subjects, where output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
#                             so sorted output_labels = [0, 2, 3, 4, 17, 41, 42, 43, 53]
# dice = [[xxx, xxx],  # scores for label 0
#         [xxx, xxx],  # scores for label 2
#         [xxx, xxx],  # scores for label 3
#         [xxx, xxx],  # scores for label 4
#         [xxx, xxx],  # scores for label 17
#         [xxx, xxx],  # scores for label 41
#         [xxx, xxx],  # scores for label 42
#         [xxx, xxx],  # scores for label 43
#         [xxx, xxx]]  # scores for label 53
#          /     \
#    subject 1  subject 2
#
# The provided GT folder must contain images sorted in the same order as images in path_images (images/gt are matched by
# sorting order).
gt_folder = '../../data/inputs/gt.nii.gz'
# Also we can compute different surface distances (Hausdorff, Hausdorff99, Hausdorff95 and mean surface distance). The
# results will be saved in arrays similar to the Dice scores.
compute_distances = True

# All right, we're ready to make predictions !!
predict(path_images,
        path_segm,
        path_model,
        path_segmentation_labels,
        n_neutral_labels=n_neutral_labels,
        path_posteriors=path_posteriors,
        path_resampled=path_resampled,
        path_volumes=path_vol,
        padding=padding,
        cropping=cropping,
        target_res=target_res,
        flip=flip,
        topology_classes=topology_classes,
        sigma_smoothing=sigma_smoothing,
        keep_biggest_component=keep_biggest_component,
        conv_size=conv_size,
        n_levels=n_levels,
        nb_conv_per_level=nb_conv_per_level,
        unet_feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier,
        activation=activation,
        gt_folder=gt_folder,
        compute_distances=compute_distances)
