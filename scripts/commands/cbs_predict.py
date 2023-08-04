import glob

import simple_parsing
import os
from dataclasses import dataclass
from typing import Optional, List, Union
from simple_parsing.helpers.serialization import Serializable

from SynthSeg.training_options import TrainingOptions
from SynthSeg.predict import predict


@dataclass
class PredictOptions(Serializable):
    training_config: Optional[str] = None
    """
    Path to the configuration file used in training the model. This is important because it will specify all
    network parameters that are also required during the prediction process.
    """

    path_images: Optional[str] = None
    """
    Path of the images to segment. Can be the path to a directory or the path to a single image.
    """

    path_segmentations: Optional[str] = None
    """
    Path where segmentations will be written.
    Should be a directory, if path_images is a directory, and a file if path_images is a file.
    """

    path_model: Optional[str] = None
    """
    Path ot the trained model. If none, the model from within the configuration directory is used.
    """

    names_segmentation: Optional[List[str]] = None
    """
    List of names corresponding to the names of the segmentation labels.
    Only used when path_volumes is provided. Must be of the same size as segmentation_labels.
    """

    path_posteriors: Optional[str] = None
    """
    Path where posteriors will be written.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    """

    path_resampled: Optional[str] = None
    """
    Path where images resampled to 1mm isotropic will be written.
    We emphasise that images are resampled as soon as the resolution in one of the axes is not in the range [0.9; 1.1].
    Should be a dir, if path_images is a dir, and a file if path_images is a file. Default is None, where resampled
    images are not saved.
    """

    path_volumes: Optional[str] = None
    """
    Path of a CSV file where the soft volumes of all segmented regions will be written.
    The rows of the CSV file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    """

    min_pad: Union[None, int, List[int], str] = None
    """
    Minimum size of the images to process. Can be an int, a sequence or a 1d numpy array.
    """

    cropping: Optional[List[int]] = None
    """
    Crop the images to the specified shape before predicting the segmentation maps.
    """

    target_res: float = 1.
    """
    Target resolution at which the network operates (and thus resolution of the output
    segmentations). This must match the resolution of the training data ! target_res is used to automatically resampled
    the images with resolutions outside [target_res-0.05, target_res+0.05].
    """

    gradients: Union[None, List[int], str] = None
    """
    Whether to replace the image by the magnitude of its gradient as input to the network.
    Can be a sequence, a 1d numpy array. Set to None to disable the automatic resampling. Default is 1mm.
    """

    flip: bool = True
    """
    Whether to perform test-time augmentation, where the input image is segmented along with
    a right/left flipped version on it. If set to True (default), be careful because this requires more memory.
    """

    topology_classes: Union[None, List[int], str] = None
    """
    List of classes corresponding to all segmentation labels, in order to group them into
    classes, for each of which we will operate a smooth version of biggest connected component.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array in the same order as segmentation_labels.
    Default is None, where no topological analysis is performed.
    """

    sigma_smoothing = 0.5
    """
    If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    """

    keep_biggest_component: bool = True
    """
    Whether to only keep the biggest component in the predicted segmentation.
    This is applied independently of topology_classes, and it is applied to the whole segmentation
    """

    gt_folder: Optional[str] = None
    """
    Path of the ground truth label maps corresponding to the input images. Should be a dir,
    if path_images is a dir, or a file if path_images is a file.
    """

    evaluation_labels: Union[None, List[int], str] = None
    """
    if gt_folder is True you can evaluate the Dice scores on a subset of the
    segmentation labels, by providing another label list here. Can be a sequence, a 1d numpy array, or the path to a
    numpy 1d array. Default is np.unique(segmentation_labels).
    """

    list_incorrect_labels: Union[None, List[int], str] = None
    """
    This option enables to replace some label values in the obtained
    segmentations by other label values. Can be a list, a 1d numpy array, or the path to such an array.
    """

    list_correct_labels: Union[None, List[int], str] = None
    """
    List of values to correct the labels specified in list_incorrect_labels.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    """
    compute_distances: bool = False
    """
    Whether to add Hausdorff and mean surface distance evaluations to the default
    Dice evaluation.
    """

    recompute: bool = True
    """
    Whether to recompute segmentations that were already computed. This also applies to
    Dice scores, if gt_folder is not None. Default is True.
    """

    verbose: bool = True
    """
    Whether to print out info about the remaining number of cases.
    """


if __name__ == '__main__':
    parser = simple_parsing.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(PredictOptions, dest="config")
    predict_options: PredictOptions = parser.parse_args().config
    file_name = predict_options.training_config

    if file_name is None:
        print("Missing training configuration file necessary for prediction")
        exit(1)

    # Check if the configuration file exists and load it
    if (not os.path.isfile(file_name)) or (not file_name.endswith(".json")) or (not os.access(file_name, os.R_OK)):
        raise RuntimeError(f"Configuration file {file_name} does not exist or is not readable.")

    # Loading the training options and fixing all relative paths
    training_options = TrainingOptions.load(file_name)
    training_options = training_options.with_absolute_paths(os.path.abspath(file_name))

    if predict_options.path_model is None:
        model_files = glob.glob(f"{os.path.dirname(os.path.abspath(file_name))}/*.h5")
        if len(model_files) > 0:
            predict_options.path_model = model_files[0]
        else:
            raise RuntimeError("No SynthSeg model specified and none available in the training config folder.")

    if predict_options.path_images is None:
        raise RuntimeError("Option path_images: No input image or directory with input images is provided")

    if predict_options.path_segmentations is None:
        raise RuntimeError("Option path_segmentations: No output directory for the segmentations is provided")

    predict(predict_options.path_images,
            predict_options.path_segmentations,
            predict_options.path_model,
            training_options.segmentation_labels,
            n_neutral_labels=training_options.n_neutral_labels,
            names_segmentation=predict_options.names_segmentation,
            path_posteriors=predict_options.path_posteriors,
            path_resampled=predict_options.path_resampled,
            path_volumes=predict_options.path_volumes,
            min_pad=predict_options.min_pad,
            cropping=predict_options.cropping,
            target_res=predict_options.target_res,
            gradients=predict_options.gradients,
            flip=predict_options.flip,
            topology_classes=predict_options.topology_classes,
            sigma_smoothing=predict_options.sigma_smoothing,
            keep_biggest_component=predict_options.keep_biggest_component,
            n_levels=training_options.n_levels,
            nb_conv_per_level=training_options.nb_conv_per_level,
            conv_size=training_options.conv_size,
            unet_feat_count=training_options.unet_feat_count,
            feat_multiplier=training_options.feat_multiplier,
            activation=training_options.activation,
            gt_folder=predict_options.gt_folder,
            evaluation_labels=predict_options.evaluation_labels,
            list_incorrect_labels=predict_options.list_incorrect_labels,
            list_correct_labels=predict_options.list_correct_labels,
            compute_distances=predict_options.compute_distances,
            recompute=predict_options.recompute,
            verbose=predict_options.verbose)
