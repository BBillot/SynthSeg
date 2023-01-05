"""

Examples to show how to estimate of the hyperparameters governing the GMM prior distributions.
This in the case where you want to train contrast-specific versions of SynthSeg.
Beware, if you do so, your model will not be able to segment any contrast at test time !
We do not provide example images and associated label maps, so do not try to run this directly !




If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


from SynthSeg.estimate_priors import build_intensity_stats

# ----------------------------------------------- simple uni-modal case ------------------------------------------------

# paths of directories containing the images and corresponding label maps
image_dir = '/image_folder/t1'
labels_dir = '/labels_folder'
# list of labels from which we want to evaluate the GMM prior distributions
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
# path of folder where to write estimated priors
result_dir = './outputs_tutorial_6/t1_priors'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      result_dir=result_dir,
                      rescale=True)

# ------------------------------------ building Gaussian priors from several labels ------------------------------------

# same as before
image_dir = '/image_folder/t1'
labels_dir = '/labels_folder'
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
result_dir = './outputs_tutorial_6/estimated_t1_priors_classes'

# In the previous example, each label value is used to build the priors of a single Gaussian distribution.
# We show here how to build Gaussian priors from intensities associated to several label values. For example, that could
# be building the Gaussian prior of white matter by using the labels of right and left white matter.
# This is done by specifying a vector, which regroups label values into "classes".
# Labels sharing the same class will contribute to the construction of the same Gaussian prior.
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)

# ---------------------------------------------- simple multi-modal case -----------------------------------------------

# Here we have multi-modal images, where every image contains all channels.
# Channels are supposed to be sorted in the same order for all subjects.
image_dir = '/image_folder/multi-modal_t1_t2'

# same as before
labels_dir = '/labels_folder'
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'
result_dir = './outputs_tutorial_6/estimated_priors_multi_modal'

build_intensity_stats(list_image_dir=image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)

# -------------------------------------  multi-modal images with separate channels -------------------------------------

# Here we have multi-modal images, where the different channels are stored in separate directories.
# We provide the these different directories as a list.
list_image_dir = ['/image_folder/t1', '/image_folder/t2']
# In this example, we assume that channels are registered and at the same resolutions.
# Therefore we can use the same label maps for all channels.
labels_dir = '/labels_folder'

# same as before
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'
result_dir = './outputs_tutorial_6/estimated_priors_multi_modal'

build_intensity_stats(list_image_dir=list_image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)

# ------------------------------------ multi-modal case with unregistered channels -------------------------------------

# Again, we have multi-modal images where the different channels are stored in separate directories.
list_image_dir = ['/image_folder/t1', '/image_folder/t2']
# In this example, we assume that the channels are no longer registered.
# Therefore we cannot use the same label maps for all channels, and must provide label maps for all modalities.
labels_dir = ['/labels_folder/t1', '/labels_folder/t2']

# same as before
estimation_labels = '../../data/labels_classes_priors/generation_labels.npy'
estimation_classes = '../../data/labels_classes_priors/generation_classes.npy'
result_dir = './outputs_tutorial_6/estimated_unregistered_multi_modal'

build_intensity_stats(list_image_dir=list_image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)
