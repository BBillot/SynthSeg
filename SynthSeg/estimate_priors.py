# python imports
import os
import numpy as np
from scipy.stats import median_absolute_deviation

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def estimate_t2_cropping(image_dir, result_dir=None, dilation=5):
    """This function takes all the hippocampus images (with 2 channels) within the specified directory, and estimates
    the cropping dimensions around the hippocampus in the t2 channel.
    It returns the mean and sts deviation for the minimal and maximal croppings, proportional to image size.
    :param image_dir: path of the folder containing hippocampus images
    :param result_dir: if not None, path of the folder where to write the computed statistics.
    :param dilation: dilation coefficient used to extract full brain mask. Default is 5.
    :returns t2_cropping_stats: numpy vector of size 4 [mean min crop, std min crop, mean max crop, std max crop]
    """

    # create result dir
    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

    # loop through images
    list_image_paths = utils.list_images_in_folder(image_dir)
    max_cropping_proportions = np.zeros(len(list_image_paths))
    min_cropping_proportions = np.zeros(len(list_image_paths))
    for im_idx, image_path in enumerate(list_image_paths):
        utils.print_loop_info(im_idx, len(list_image_paths), 10)

        # load t2 channel
        im = utils.load_volume(image_path)
        t2 = im[..., 1]
        shape = t2.shape
        hdim = int(np.argmax(shape))

        # mask image
        _, mask = edit_volumes.mask_volume(t2, threshold=0, dilate=dilation, return_mask=True)

        # find cropping indices
        indices = np.nonzero(mask)[hdim]
        min_cropping_proportions[im_idx] = np.maximum(np.min(indices) + int(dilation/2), 0) / shape[hdim]
        max_cropping_proportions[im_idx] = np.minimum(np.max(indices) - int(dilation/2), shape[hdim]) / shape[hdim]

    # compute and save stats
    t2_cropping_stats = np.array([np.mean(min_cropping_proportions),
                                  np.std(min_cropping_proportions),
                                  np.mean(max_cropping_proportions),
                                  np.std(max_cropping_proportions)])

    # save stats if necessary
    if result_dir is not None:
        np.save(os.path.join(result_dir, 't2_cropping_stats.npy'), t2_cropping_stats)

    return t2_cropping_stats


def sample_intensity_stats_from_image(image, segmentation, labels_list, classes_list=None, keep_strictly_positive=True):
    """This function takes an image and corresponding segmentation as inputs. It estimates the mean and std intensity
    for all specified label values. Labels can share the same statistics by being regrouped into classes.
    :param image: image from which to evaluate mean intensity and std deviation.
    :param segmentation: segmentation of the input image. Must have the same size as image.
    :param labels_list: list of labels for which to evaluate mean and std intensity.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    :param classes_list: (optional) enables to regroup structures into classes of similar intensity statistics.
    Can be a sequence or a 1d numpy array, or the path to a 1d numpy array, of the same length as label_list.
    :param keep_strictly_positive: (optional) whether to only keep strictly positive intensity values when
    computing stats. This doesn't apply to the first label in label_list (or class if class_list is provided), for
    which we keep positive and zero values, as we consider it to be the background label.
    :return: a numpy array of size (2, len(label_list)), the first row being the mean intenisty for each structure,
    and the second being the median absolute deviation (robust estimation of std).
    """

    # reformat labels and classes
    labels_list = np.array(utils.reformat_to_list(labels_list, load_as_numpy=True, dtype='int'))
    if classes_list is not None:
        classes_list = np.array(utils.reformat_to_list(classes_list, load_as_numpy=True, dtype='int'))
    else:
        classes_list = np.arange(labels_list.shape[0])
    assert len(classes_list) == len(labels_list), 'labels and classes lists should have the same length'

    # compute mean/std of specified classes
    means = np.zeros(labels_list.shape[0])
    stds = np.zeros(labels_list.shape[0])
    unique_classes, unique_indices = np.unique(classes_list, return_index=True)
    for idx, tmp_class in enumerate(unique_classes):

        # get list of all intensity values for the current class
        class_labels = labels_list[classes_list == tmp_class]
        intensities = np.empty(0)
        for label in class_labels:
            tmp_intensities = image[segmentation == label]
            intensities = np.concatenate([intensities, tmp_intensities])
        if tmp_class:  # i.e. if not background
            if keep_strictly_positive:
                intensities = intensities[intensities > 0]

        # compute stats for class and put them to the location of corresponding label values
        if len(intensities) != 0:
            means[classes_list == tmp_class] = np.nanmedian(intensities)
            stds[classes_list == tmp_class] = median_absolute_deviation(intensities, nan_policy='omit')

    return np.stack([means, stds])


def sample_intensity_stats_from_dataset(image_dir, labels_dir, labels_list, classes_list=None, rescale=True):
    """This function estimates intensity distributions on real images of the same dataset.
    It returns a normal distribution for the mean intensity of each label, and a normal distribution for the intensity
    standard deviation of each label. Thus each label value is associated to 4 parameters: a mean/std for the mean
    intensity, and a mean/std for the std deviation.
    Labels can share the same statistics by being regrouped into classes.
    :param image_dir: path of directory with images to estimate the intensity distribution
    :param labels_dir: path of directory with segmentation of input images.
    They are matched with images by sorting order.
    :param labels_list: list of labels for which to evaluate mean and std intensity.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    :param classes_list: (optional) enables to regroup structures into classes of similar intensity statistics.
    Can be a sequence or a 1d numpy array, or the path to a 1d numpy array, of the same length as label_list.
    :param rescale: (optional) whether to rescale images between 0 and 255 before intensity estimation
    :return: 2 numpy arrays of size (2, len(label_list)), one with the evaluated means/std for the mean
    intensity, and one for the mean/std for the standard deviation.
    For each array, the first row represents the mean, and the second represents the std.
    """

    # list files
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), 'image and labels folders do not have the same number of files'

    # initialise result arrays
    labels_list = np.array(utils.reformat_to_list(labels_list, load_as_numpy=True, dtype='int'))
    n_dims, n_channels = utils.get_dims(utils.load_volume(path_images[0]).shape, max_channels=3)
    means = np.zeros((len(path_images), labels_list.shape[0], n_channels))
    stds = np.zeros((len(path_images), labels_list.shape[0], n_channels))

    # loop over images
    for idx, (path_im, path_la) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load image and label map
        image = utils.load_volume(path_im)
        la = utils.load_volume(path_la)
        if n_channels == 1:
            image = utils.add_axis(image, -1)

        # loop over channels
        for channel in range(n_channels):
            im = image[..., channel]
            if rescale:
                im = edit_volumes.rescale_volume(im)
            stats = sample_intensity_stats_from_image(im, la, labels_list, classes_list=classes_list)
            means[idx, :, channel] = stats[0, :]
            stds[idx, :, channel] = stats[1, :]

    # compute prior parameters for mean/std
    mean_means = np.mean(means, axis=0)
    std_means = np.std(means, axis=0)
    mean_stds = np.mean(stds, axis=0)
    std_stds = np.std(stds, axis=0)

    # regroup prior parameters in two different arrays: one for the mean and one for the std
    estimated_means = np.zeros((2*n_channels, len(labels_list)))
    estimated_stds = np.zeros((2*n_channels, len(labels_list)))
    for channel in range(n_channels):
        estimated_means[2*channel, :] = mean_means[:, channel]
        estimated_means[2 * channel + 1, :] = std_means[:, channel]
        estimated_stds[2*channel, :] = mean_stds[:, channel]
        estimated_stds[2 * channel + 1, :] = std_stds[:, channel]

    return estimated_means, estimated_stds


def build_intensity_stats_for_several_modalities(list_image_dir,
                                                 list_labels_dir,
                                                 estimation_labels,
                                                 estimation_classes,
                                                 generation_classes,
                                                 results_dir,
                                                 rescale=True):
    """This function aims at estimating the intensity distributions of K different structures from a set of images.
    The distribution of each structure is modelled as a Gaussian, thus parametrised by a mean and a standard deviation.
    Because the intensity distribution of structures can vary accross images, we additionally use Gausian priors for the
    parameters of each Gaussian distribution. Therefore, the intensity distribution of each structure is described by 4
    parameters (a mean/std for the mean intensity, and a mean/std for the std deviation).
    This function uses a set of images along with corresponding segmentations to estimate the 4*K parameters.
    Additionally, it can estimate the 4*K parameters for several datasets of images, that we call here n_datasets.
    This function writes 2 numpy arrays of size (2*n_datasets, K), one with the evaluated means/std for the mean
    intensity, and one for the mean/std for the standard deviation.
    In these arrays, each block of two rows refer to a different modality.
    Within each block of two rows, the first row represents the mean, and the second represents the std.
    :param list_image_dir: path of folders with images for intensity distribution estimation.
    Can be the path of single directory (n_datasets=1), or a list of folders, each being a separate dataset.
    Images can also be directly multimodal, in which case each modality is treated as a different dataset, i.e. each
    modality will have a separate block (of size (2, K)) in the result arrays.
    :param list_labels_dir: path of folders with label maps corresponding to input images.
    If list_image_dir is a list of several folders, list_labels_dir can either be a list of folders, or the path to a
    single folder, which will be used for all datasets.
    If a dataset has multi-modal images, the same label map is applied to all modalities.
    :param estimation_labels: labels to sample intensity from.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    :param estimation_classes: Indices regrouping labels into classes when estimating intensity distribution parameters.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array. Must have the same length as estimation_labels
    :param generation_classes: Indices regrouping into classes the label values within the generation label maps.
    Generation labels with the same class will share the same estimated priors when generating a new image.
    Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
    :param results_dir: path of directory where estimated priors will be writen.
    :param rescale: (optional) whether to rescale images between 0 and 255 before intensity estimation
    """

    # reformat variables into lists
    list_image_dir = utils.reformat_to_list(list_image_dir)
    list_labels_dir = utils.reformat_to_list(list_labels_dir, length=len(list_image_dir))

    # handle results directories
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # loop over dataset
    list_means = list()
    list_stds = list()
    for image_dir, labels_dir in zip(list_image_dir, list_labels_dir):

        # get prior stats for dataset
        estimated_means, estimated_stds = sample_intensity_stats_from_dataset(image_dir,
                                                                              labels_dir,
                                                                              estimation_labels,
                                                                              estimation_classes,
                                                                              rescale=rescale)

        # get unique stats (only one per class)
        estimation_labels = np.array(
            utils.reformat_to_list(estimation_labels, load_as_numpy=True, dtype='int'))
        if estimation_classes is not None:
            estimation_classes = np.array(
                utils.reformat_to_list(estimation_classes, load_as_numpy=True, dtype='int'))
        else:
            estimation_classes = np.arange(estimation_labels.shape[0])
        unique_estimation_classes, unique_indices = np.unique(estimation_classes, return_index=True)
        estimated_means = estimated_means[:, unique_indices]
        estimated_stds = estimated_stds[:, unique_indices]

        # get unique generation classes
        generation_classes = np.array(utils.reformat_to_list(generation_classes, load_as_numpy=True, dtype='int'))
        unique_generation_classes = np.unique(generation_classes)
        missing = [v for v in unique_generation_classes if v not in unique_estimation_classes]
        assert missing == [], 'stats for generation classes {} cannot be computed, because they are not in ' \
                              'estimation classes'.format(missing)

        # reorder estimated means/stds in the order specified in generation classes
        means = np.zeros((2, len(generation_classes)))
        stds = np.zeros((2, len(generation_classes)))
        for idx, tmp_class in enumerate(unique_generation_classes):
            means[:, generation_classes == tmp_class] = estimated_means[:, unique_estimation_classes == tmp_class]
            stds[:, generation_classes == tmp_class] = estimated_stds[:, unique_estimation_classes == tmp_class]

        list_means.append(means)
        list_stds.append(stds)

    # stack all modalities together
    means_range = np.concatenate(list_means, axis=0)
    stds_range = np.concatenate(list_stds, axis=0)

    # save files
    np.save(os.path.join(results_dir, 'means_range.npy'), means_range)
    np.save(os.path.join(results_dir, 'std_devs_range.npy'), stds_range)
