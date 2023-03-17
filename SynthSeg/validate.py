"""
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


# python imports
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

# project imports
from SynthSeg.predict import predict

# third-party imports
from ext.lab2im import utils


def validate_training(image_dir,
                      gt_dir,
                      models_dir,
                      validation_main_dir,
                      labels_segmentation,
                      n_neutral_labels=None,
                      evaluation_labels=None,
                      step_eval=1,
                      min_pad=None,
                      cropping=None,
                      target_res=1.,
                      gradients=False,
                      flip=False,
                      topology_classes=None,
                      sigma_smoothing=0,
                      keep_biggest_component=False,
                      n_levels=5,
                      nb_conv_per_level=2,
                      conv_size=3,
                      unet_feat_count=24,
                      feat_multiplier=2,
                      activation='elu',
                      recompute=False):
    """This function validates models saved at different epochs of the same training.
    All models are assumed to be in the same folder.
    The results of each model are saved in a subfolder in validation_main_dir.
    :param image_dir: path of the folder with validation images.
    :param gt_dir: path of the folder with ground truth label maps.
    These are matched to the validation images by sorting order.
    :param models_dir: path of the folder with the models to validate.
    :param validation_main_dir: path of the folder where all the models validation subfolders will be saved.
    :param labels_segmentation: path of the numpy array containing all the segmentation labels used during training.
    :param n_neutral_labels: (optional) value of n_neutral_labels used during training. Used only if flip is True.
    :param evaluation_labels: (optional) label values to validate on. Must be a subset of the segmentation labels.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array. Default is the same as segmentation_label_list
    :param step_eval: (optional) If step_eval > 1 skips models when validating, by validating on models step_eval apart.
    :param cropping: (optional) whether to crop the input to smaller size while being run through the network.
    The result is then given in the original image space. Can be an int, a sequence, or a 1d numpy array.
    :param target_res: (optional) target resolution at which the network operates (and thus resolution of the output
    segmentations). This must match the resolution of the training data ! target_res is used to automatically resampled
    the images with resolutions outside [target_res-0.05, target_res+0.05].
    Can be a sequence, a 1d numpy array. Set to None to disable the automatic resampling. Default is 1mm.
    :param flip: (optional) whether to perform test-time augmentation, where the input image is segmented along with
    a right/left flipped version on it. If set to True (default), be careful because this requires more memory.
    :param topology_classes: List of classes corresponding to all segmentation labels, in order to group them into
    classes, for each of which we will operate a smooth version of biggest connected component.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array in the same order as segmentation_label_list.
    Default is None, where no topological analysis is performed.
    :param sigma_smoothing: (optional) If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    :param keep_biggest_component: (optional) whether to only keep the biggest component in the predicted segmentation.
    This is applied independently of topology_classes, and it is applied to the whole segmentation
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature maps for the first level. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this number at each new level. Default is 1.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.
    :param recompute: (optional) whether to recompute result files even if they already exist."""

    # create result folder
    utils.mkdir(validation_main_dir)

    # loop over models
    list_models = utils.list_files(models_dir, expr=['dice', '.h5'], cond_type='and')[::step_eval]
    # list_models = [p for p in list_models if int(os.path.basename(p)[-6:-3]) % 10 == 0]
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        # build names and create folders
        model_val_dir = os.path.join(validation_main_dir, os.path.basename(path_model).replace('.h5', ''))
        dice_path = os.path.join(model_val_dir, 'dice.npy')
        utils.mkdir(model_val_dir)

        if (not os.path.isfile(dice_path)) | recompute:
            loop_info.update(model_idx)
            predict(path_images=image_dir,
                    path_segmentations=model_val_dir,
                    path_model=path_model,
                    labels_segmentation=labels_segmentation,
                    n_neutral_labels=n_neutral_labels,
                    min_pad=min_pad,
                    cropping=cropping,
                    target_res=target_res,
                    gradients=gradients,
                    flip=flip,
                    topology_classes=topology_classes,
                    sigma_smoothing=sigma_smoothing,
                    keep_biggest_component=keep_biggest_component,
                    n_levels=n_levels,
                    nb_conv_per_level=nb_conv_per_level,
                    conv_size=conv_size,
                    unet_feat_count=unet_feat_count,
                    feat_multiplier=feat_multiplier,
                    activation=activation,
                    gt_folder=gt_dir,
                    evaluation_labels=evaluation_labels,
                    recompute=recompute,
                    verbose=False)


def plot_validation_curves(list_validation_dirs, architecture_names=None, eval_indices=None,
                           skip_first_dice_row=True, size_max_circle=100, figsize=(11, 6), y_lim=None, fontsize=18,
                           list_linestyles=None, list_colours=None, plot_legend=False, draw_line=None):
    """This function plots the validation curves of several networks, based on the results of validate_training().
    It takes as input a list of validation folders (one for each network), each containing subfolders with dice scores
    for the corresponding validated epoch.
    :param list_validation_dirs: list of all the validation folders of the trainings to plot.
    :param eval_indices: (optional) compute the average Dice on a subset of labels indicated by the specified indices.
    Can be a 1d numpy array, the path to such an array, or a list of 1d numpy arrays as long as list_validation_dirs.
    :param skip_first_dice_row: if eval_indices is None, skip the first row of the dice matrices (usually background)
    :param size_max_circle: (optional) size of the marker for epochs achieving the best validation scores.
    :param figsize: (optional) size of the figure to draw.
    :param fontsize: (optional) fontsize used for the graph."""

    n_curves = len(list_validation_dirs)

    if eval_indices is not None:
        if isinstance(eval_indices, (np.ndarray, str)):
            if isinstance(eval_indices, str):
                eval_indices = np.load(eval_indices)
            eval_indices = np.squeeze(utils.reformat_to_n_channels_array(eval_indices, n_dims=len(eval_indices)))
            eval_indices = [eval_indices] * len(list_validation_dirs)
        elif isinstance(eval_indices, list):
            for (i, e) in enumerate(eval_indices):
                if isinstance(e, np.ndarray):
                    eval_indices[i] = np.squeeze(utils.reformat_to_n_channels_array(e, n_dims=len(e)))
                else:
                    raise TypeError('if provided as a list, eval_indices should only contain numpy arrays')
        else:
            raise TypeError('eval_indices can be a numpy array, a path to a numpy array, or a list of numpy arrays.')
    else:
        eval_indices = [None] * len(list_validation_dirs)

    # reformat model names
    if architecture_names is None:
        architecture_names = [os.path.basename(os.path.dirname(d)) for d in list_validation_dirs]
    else:
        architecture_names = utils.reformat_to_list(architecture_names, len(list_validation_dirs))

    # prepare legend labels
    if plot_legend is False:
        list_legend_labels = ['_nolegend_'] * n_curves
    elif plot_legend is True:
        list_legend_labels = architecture_names
    else:
        list_legend_labels = architecture_names
        list_legend_labels = ['_nolegend_' if i >= plot_legend else list_legend_labels[i] for i in range(n_curves)]

    # prepare linestyles
    if list_linestyles is not None:
        list_linestyles = utils.reformat_to_list(list_linestyles)
    else:
        list_linestyles = [None] * n_curves

    # prepare curve colours
    if list_colours is not None:
        list_colours = utils.reformat_to_list(list_colours)
    else:
        list_colours = [None] * n_curves

    # loop over architectures
    plt.figure(figsize=figsize)
    for idx, (net_val_dir, net_name, linestyle, colour, legend_label, eval_idx) in enumerate(zip(list_validation_dirs,
                                                                                                 architecture_names,
                                                                                                 list_linestyles,
                                                                                                 list_colours,
                                                                                                 list_legend_labels,
                                                                                                 eval_indices)):

        list_epochs_dir = utils.list_subfolders(net_val_dir, whole_path=False)

        # loop over epochs
        list_net_scores = list()
        list_epochs = list()
        for epoch_dir in list_epochs_dir:

            # build names and create folders
            path_epoch_scores = os.path.join(net_val_dir, epoch_dir, 'dice.npy')
            if os.path.isfile(path_epoch_scores):
                if eval_idx is not None:
                    list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores)[eval_idx, :])))
                else:
                    if skip_first_dice_row:
                        list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores)[1:, :])))
                    else:
                        list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores))))
                list_epochs.append(int(re.sub('[^0-9]', '', epoch_dir)))

        # plot validation scores for current architecture
        if list_net_scores:  # check that archi has been validated for at least 1 epoch
            list_net_scores = np.array(list_net_scores)
            list_epochs = np.array(list_epochs)
            list_epochs, idx = np.unique(list_epochs, return_index=True)
            list_net_scores = list_net_scores[idx]
            max_score = np.max(list_net_scores)
            epoch_max_score = list_epochs[np.argmax(list_net_scores)]
            print('\n'+net_name)
            print('epoch max score: %d' % epoch_max_score)
            print('max score: %0.3f' % max_score)
            plt.plot(list_epochs, list_net_scores, label=legend_label, linestyle=linestyle, color=colour)
            plt.scatter(epoch_max_score, max_score, s=size_max_circle, color=colour)

    # finalise plot
    plt.grid()
    if draw_line is not None:
        draw_line = utils.reformat_to_list(draw_line)
        list_linestyles = ['dotted', 'dashed', 'solid', 'dashdot'][:len(draw_line)]
        for line, linestyle in zip(draw_line, list_linestyles):
            plt.axhline(line, color='black', linestyle=linestyle)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)  # set right/left limits of plot
    plt.title('Validation curves', fontsize=fontsize)
    if plot_legend:
        plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()


def draw_learning_curve(path_tensorboard_files, architecture_names, figsize=(11, 6), fontsize=18,
                        y_lim=None, remove_legend=False):
    """This function draws the learning curve of several trainings on the same graph.
    :param path_tensorboard_files: list of tensorboard files corresponding to the models to plot.
    :param architecture_names: list of the names of the models
    :param figsize: (optional) size of the figure to draw.
    :param fontsize: (optional) fontsize used for the graph.
    """

    # reformat inputs
    path_tensorboard_files = utils.reformat_to_list(path_tensorboard_files)
    architecture_names = utils.reformat_to_list(architecture_names)
    assert len(path_tensorboard_files) == len(architecture_names), 'names and tensorboard lists should have same length'

    # loop over architectures
    plt.figure(figsize=figsize)
    for path_tensorboard_file, name in zip(path_tensorboard_files, architecture_names):

        path_tensorboard_file = utils.reformat_to_list(path_tensorboard_file)

        # extract loss at the end of all epochs
        list_losses = list()
        list_epochs = list()
        logging.getLogger('tensorflow').disabled = True
        for path in path_tensorboard_file:
            for e in summary_iterator(path):
                for v in e.summary.value:
                    if v.tag == 'loss' or v.tag == 'accuracy' or v.tag == 'epoch_loss':
                        list_losses.append(v.simple_value)
                        list_epochs.append(e.step)
        plt.plot(np.array(list_epochs), 1-np.array(list_losses), label=name, linewidth=2)

    # finalise plot
    plt.grid()
    if not remove_legend:
        plt.legend(fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)  # set right/left limits of plot
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.title('Learning curves', fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()
