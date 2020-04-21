# python imports
import os
import warnings
import numpy as np
from scipy.stats import wilcoxon
from scipy.ndimage.morphology import distance_transform_edt

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def fast_dice(x, y, labels):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on, sorted in increasing order.
    :return: numpy array with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    label_edges = np.concatenate([labels[0:1] - 0.5, labels + 0.5])
    hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)
    c = hst[0]

    return np.diag(c) * 2 / (np.sum(c, 0) + np.sum(c, 1) + 1e-5)


def surface_distances(x, y):
    """Computes the maximum boundary distance (Haussdorf distance), and the average boundary distance of two masks.
    x and y should be boolean or 0/1 numpy arrays of the same size."""

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    # detect edge
    x_dist_int = distance_transform_edt(x * 1)
    x_edge = (x_dist_int == 1) * 1
    y_dist_int = distance_transform_edt(y * 1)
    y_edge = (y_dist_int == 1) * 1

    # calculate distance from edge
    x_dist = distance_transform_edt(np.logical_not(x_edge))
    y_dist = distance_transform_edt(np.logical_not(y_edge))

    # find distances from the 2 surfaces
    x_dists_to_y = y_dist[x_edge == 1]
    y_dists_to_x = x_dist[y_edge == 1]

    # find max distance from the 2 surfaces
    x_max_dist_to_y = np.max(x_dists_to_y)
    y_max_dist_to_x = np.max(y_dists_to_x)

    # find average distance between 2 surfaces
    x_mean_dist_to_y = np.mean(x_dists_to_y)
    y_mean_dist_to_x = np.mean(y_dists_to_x)

    return np.maximum(x_max_dist_to_y, y_max_dist_to_x), (x_mean_dist_to_y + y_mean_dist_to_x) / 2


def compute_non_parametric_paired_test(dice_ref, dice_compare, eval_indices, alternative='two-sided'):
    """Compute non-parametric paired t-tests between two sets of Dice scores.
    :param dice_ref: numpy array with Dice scores, rows represent structures, and columns represent subjects.
    Taken as reference for one-sided tests.
    :param dice_compare: numpy array of the same format as dice_ref.
    :param eval_indices: (optional) list or 1d array indicating the row indices of structures to run the tests for
    :param alternative: (optional) The alternative hypothesis to be tested, Cab be 'two-sided', 'greater', 'less'.
    :return: 1d numpy array, with p-values for all tests on evaluated structures, as well as an additionnal test for
    average scores (last value of the array). The average score is computed only on the evaluation structures.
    """

    # loop over all evaluation label values
    pvalues = list()
    for idx in eval_indices:

        x = dice_ref[idx, :]
        y = dice_compare[idx, :]
        _, p = wilcoxon(x, y, alternative=alternative)
        pvalues.append(p)

    # average score
    x = np.mean(dice_ref[eval_indices, :], axis=0)
    y = np.mean(dice_compare[eval_indices, :], axis=0)
    _, p = wilcoxon(x, y, alternative=alternative)
    pvalues.append(p)

    return np.array(pvalues)


def dice_evaluation(gt_folder,
                    seg_folder,
                    path_segmentation_label_list,
                    path_result_dice_array):
    """Computes Dice scores for all labels contained in path_segmentation_label_list. Files in gt_folder and seg_folder
    are matched by sorting order.
    :param gt_folder: folder containing ground truth files.
    :param seg_folder: folder containing evaluation files.
    :param path_segmentation_label_list: path of numpy vector containing all labels to compute the Dice for.
    :param path_result_dice_array: path where the resulting Dice will be writen as numpy array.
    :return: numpy array containing all dice scores (labels in rows, subjects in columns).
    """

    # get list of automated and manual segmentations
    list_path_gt_labels = utils.list_images_in_folder(gt_folder)
    list_path_segs = utils.list_images_in_folder(seg_folder)
    if len(list_path_gt_labels) != len(list_path_segs):
        warnings.warn('both data folders should have the same length, had {} and {}'.format(len(list_path_gt_labels),
                                                                                            len(list_path_segs)))

    # load labels list
    label_list, neutral_labels = utils.get_list_labels(label_list=path_segmentation_label_list, FS_sort=True,
                                                       labels_dir=gt_folder)

    # create result folder
    if not os.path.exists(os.path.dirname(path_result_dice_array)):
        os.mkdir(os.path.dirname(path_result_dice_array))

    # initialise result matrix
    dice_coefs = np.zeros((label_list.shape[0], len(list_path_segs)))

    # start analysis
    for im_idx, (path_gt, path_seg) in enumerate(zip(list_path_gt_labels, list_path_segs)):
        utils.print_loop_info(im_idx, len(list_path_segs), 10)

        # load gt labels and segmentation
        gt_labels = utils.load_volume(path_gt, dtype='int')
        seg = utils.load_volume(path_seg, dtype='int')
        n_dims = len(gt_labels.shape)
        # crop images
        gt_labels, cropping = edit_volumes.crop_volume_around_region(gt_labels, margin=10)
        if n_dims == 2:
            seg = seg[cropping[0]:cropping[2], cropping[1]:cropping[3]]
        elif n_dims == 3:
            seg = seg[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5]]
        else:
            raise Exception('cannot evaluate images with more than 3 dimensions')
        # extract list of unique labels
        label_list_sorted = np.sort(label_list)
        tmp_dice = fast_dice(gt_labels, seg, label_list_sorted)
        dice_coefs[:, im_idx] = tmp_dice[np.searchsorted(label_list_sorted, label_list)]

    # write dice results
    np.save(path_result_dice_array, dice_coefs)

    return dice_coefs
