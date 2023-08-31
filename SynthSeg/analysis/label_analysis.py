import nibabel as nib
import nibabel.processing as proc
import numpy as np
from os import listdir
from os.path import isfile, join
from typing import List
import re

from SynthSeg.analysis.freesurfer_tools import TissueType, generateTissueTypesFromSample


def windowRescale(nifti_file: str,
                  out_file: str,
                  min_clip: float,
                  max_clip: float,
                  min_out: float = 0.0,
                  max_out: float = 255.0):
    """
    Windows and rescales the values in a NIfTI image to a specified range.
    This function helps if you want to prepare an image that contains outliers in, e.g., noisy regions.

    Args:
        nifti_file: The path to an existing NIfTI file to be rescaled.
        out_file: The path to save the rescaled NIfTI file. Directory needs to exist.
        min_clip: The minimum value to clip the data in the NIfTI file before rescaling.
        max_clip: The maximum value to clip the data in the NIfTI file before rescaling.
        min_out: The minimum value for rescaling the data. Default is 0.0.
        max_out: The maximum value for rescaling the data. Default is 255.0.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    clipped_data = np.clip(data, min_clip, max_clip)
    rescaled_data = min_out + ((clipped_data - min_clip) * (max_out - min_out)) / (max_clip - min_clip)
    nib.save(nib.Nifti1Image(rescaled_data, img.affine, img.header), out_file)


def createContrastEntries(
        scan_file: str,
        label_file: str,
        generation_labels: list,
        percent_deviation: float = 5.0) -> dict:
    """
    Create contrast entries for given scan and label files where the label image should be a segmentation
    of the scan image. The idea behind this function is that you want to calculate the contrast statistics
    for each region specified by labels in `generation_labels`.
    The function will return a range for the mean and standard deviation of each region where the size of the range is
    determined by `percent_deviation`. These values are later used for generating training data for SynthSeg where
    the distribution of gray-values within a region is determined by randomly choosing a mean value from the range of
    mean values for the region and a randomly chosen standard deviation of each region.

    Args:
        scan_file (str): The file path of the scan.
        label_file (str): The file path of the label.
        generation_labels (list): The generation labels to use.
        percent_deviation (float, optional): The percentage deviation. Defaults to 5.0.

    Returns:
        dict: With the entries "output_labels", "generation_classes", "prior_means", and "prior_stds". All are input
        arguments for SynthSeg's brain generator.
    """
    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()
    scan_img = nib.load(scan_file)
    scan_data = scan_img.get_fdata()
    resampled_labels = proc.resample_from_to(label_img, scan_img, order=0)
    # noinspection PyUnresolvedReferences
    resampled_labels_data = resampled_labels.get_fdata()
    labels = np.unique(label_data.flatten()).astype(np.int32)
    output_labels = [x if x in labels else 0 for x in generation_labels]
    generation_classes = []
    min_prior_means = []
    max_prior_means = []
    min_prior_stds = []
    max_prior_stds = []
    for label_index in range(len(generation_labels)):
        current_label = generation_labels[label_index]
        generation_classes.append(label_index)
        if current_label not in labels:
            min_prior_means.append(0.0)
            max_prior_means.append(1.0)
            min_prior_stds.append(0.0)
            max_prior_stds.append(1.0)
        else:
            mean = np.mean(scan_data[resampled_labels_data == current_label])
            std = np.std(scan_data[resampled_labels_data == current_label])
            min_prior_means.append(mean*(1.0 - percent_deviation/100.0))
            max_prior_means.append(mean*(1.0 + percent_deviation/100.0))
            min_prior_stds.append(std*(1.0 - percent_deviation/100.0)*0.001)
            max_prior_stds.append(std*(1.0 + percent_deviation/100.0)*0.001)
    return {
        "output_labels": output_labels,
        "generation_classes": generation_classes,
        "prior_means": [min_prior_means, max_prior_means],
        "prior_stds": [min_prior_stds, max_prior_stds]}


def analyseLabelScanPair(scan_file: str, label_file: str) -> dict:
    """
    Calculates region statistics and information for a given scan and segmentation image pair.
    Scan and label images are not required to have the same resolution, but they need to represent the same view.
    The label image is rescaled to the resolution of the scan image. After that, every available segmentation class
    in the label image is processed and FreeSurfer label information is added. It returns a dict containing sorted lists
    of "neutral_regions", "left_regions" and "right_regions".

    Args:
        scan_file: File path to the NIfTI scan file.
        label_file: File path to the NIfTI label file.

    Returns:
        A dictionary containing the analysis results of the label and scan pair. The dictionary has the following keys:
            - "neutral_regions": A sorted list of tissue types found in regions that are not specific to left or right.
            - "left_regions": A sorted list of tissue types found in regions specific to the left side.
            - "right_regions": A sorted list of tissue types found in regions specific to the right side.
    """
    label_img = nib.load(label_file)
    label_data = label_img.get_fdata()
    scan_img = nib.load(scan_file)
    scan_data = scan_img.get_fdata()
    resampled_labels = proc.resample_from_to(label_img, scan_img, order=0)
    # noinspection PyUnresolvedReferences
    resampled_labels_data = resampled_labels.get_fdata()
    labels = np.unique(label_data.flatten()).astype(np.int32)
    result = list(map(lambda l: generateTissueTypesFromSample(scan_data, resampled_labels_data, l), labels))

    left_pattern = re.compile("Left-.+|ctx-lh")
    right_pattern = re.compile("Right-.+|ctx-rh")

    left_regions = list(filter(lambda entry: re.match(left_pattern, entry.label.name), result))
    left_regions.sort(key=lambda entry: entry.segmentation_class)
    right_regions = list(filter(lambda entry: re.match(right_pattern, entry.label.name), result))
    right_regions.sort(key=lambda entry: entry.segmentation_class)
    neutral_regions = [reg for reg in result if reg not in left_regions and reg not in right_regions]
    neutral_regions.sort(key=lambda entry: entry.segmentation_class)
    assert len(left_regions) == len(right_regions), \
        "There should be exactly as many left regions as there are right regions"
    return {"neutral_regions": neutral_regions, "left_regions": left_regions, "right_regions": right_regions}


def equalizeLeftRightRegions(regions_dict: dict) -> dict:
    """
    Equalizes the mean and standard deviation between corresponding left and right regions.
    The input should be the dictionary that is returned by `analyseLabelScanPair`.
    The reasoning here is that when creating training data, we don't want to have different random
    distributions for corresponding left/right regions.

    Args:
        regions_dict (dict): A dictionary with keys: ['left_regions', 'neutral_regions', 'right_regions'].

    Returns:
        dict: The updated regions dictionary with equalized mean and standard deviation values.
    """
    assert sorted(list(regions_dict.keys())) == ["left_regions", "neutral_regions", "right_regions"]
    left_regions = regions_dict["left_regions"]
    right_regions = regions_dict["right_regions"]
    assert len(left_regions) == len(right_regions)
    for i in range(len(left_regions)):
        left_name = left_regions[i].label.name
        right_name = left_name.replace("Left-", "Right-").replace("ctx-lh", "ctx-rh")
        assert right_regions[i].label.name == right_name
        new_mean = 0.5*(left_regions[i].mean + right_regions[i].mean)
        new_std_dev = 0.5*(left_regions[i].std_dev + right_regions[i].std_dev)
        left_regions[i].mean = new_mean
        right_regions[i].mean = new_mean
        left_regions[i].std_dev = new_std_dev
        right_regions[i].std_dev = new_std_dev
    return regions_dict


def listAvailableLabelsInMap(nifti_file: str) -> np.ndarray:
    """
    Reads in a segmentation NIfTI image and returns a list of all labels found in the image.

    Args:

        nifti_file (str): Path to the segmentation NIfTI image

    Returns:
        np.ndarray: Sorted numpy array of all found labels
    """
    labelMap = nib.load(nifti_file)
    data = np.array(labelMap.get_data(), dtype=np.int32)
    return np.unique(data)


def findAllAvailableLabels(directory: str) -> np.ndarray:
    """
    Does the same as `listAvailableLabelsInMap` but for a whole directory containing
    segmentation maps.

    Args:
        directory (str): Directory containing the segmentation maps

    Returns:
        np.ndarray: Sorted numpy array of all available labels
    """
    result = np.array([])
    files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for f in files:
        labels = listAvailableLabelsInMap(f)
        result = np.append(result, labels)
    return np.unique(result)
