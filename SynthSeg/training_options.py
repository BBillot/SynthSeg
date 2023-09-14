from dataclasses import dataclass
from simple_parsing.helpers.serialization import Serializable

from .option_types import *
from .option_utils import get_absolute_path


@dataclass
class TrainingOptions(Serializable):
    labels_dir: str = "../../data/training_label_maps"
    """
    Path of folder with all input label maps, or to a single label map (if only one training example)
    """

    model_dir: str = "./output"
    """
    Path of a directory where the models will be saved during training.
    """

    generation_labels: Union[None, str, List[int]] = None
    """
    List of all possible label values in the input label maps.
    It can be None (default, where the label values are directly gotten from the provided label maps), a list,
    a 1d numpy array, or the path to such an array. If not None, the background label should always be the first. Then,
    if the label maps contain some right/left specific labels and if flipping is applied during training (see 'flipping'
    parameter), generation_labels should be organised as follows:
    first the background label, then the non-sided labels (i.e. those which are not right/left specific), then all the
    left labels, and finally the corresponding right labels (in the same order as the left ones). Please make sure each
    that each sided label has a right and a left value (this is essential!!!).
    :param n_neutral_labels: (optional) if the label maps contain some right/left specific labels and if flipping is
    applied during training, please provide the number of non-sided labels (including the background).
    This is used to know where the sided labels start in generation_labels. Leave to default (None) if either one of the
    two conditions is not fulfilled.
    """

    n_neutral_labels: Optional[int] = None
    """
    If the label maps contain some right/left specific labels and if flipping is
    applied during training, please provide the number of non-sided labels (including the background).
    This is used to know where the sided labels start in generation_labels. Leave to default (None) if either one of the
    two conditions is not fulfilled.
    """

    segmentation_labels: Union[None, str, List[int]] = None
    """
    List of the same length as generation_labels to indicate which values to use
    in the training label maps, i.e. all occurrences of generation_labels[i] in the input label maps will be converted
    to output_labels[i] in the returned label maps.
    Examples: Set output_labels[i] to zero if you wish to erase the value generation_labels[i] from the returned label maps.
    Set output_labels[i]=generation_labels[i] if you wish to keep the value generation_labels[i] in the returned maps.
    Can be a list or a 1d numpy array, or the path to such an array. Default is output_labels = generation_labels.    
    """

    subjects_prob: Union[None, str, List[int]] = None
    """
    Relative order of importance (doesn't have to be probabilistic), with which to pick
    the provided label maps at each minibatch. Can be a sequence, a 1D numpy array, or the path to such an array, and it
    must be as long as path_label_maps. By default, all label maps are chosen with the same importance.
    """

    # output-related parameters

    batchsize: int = 1
    """
    Number of images to generate per mini-batch.
    """

    n_channels: int = 1
    """
    number of channels to be synthesised.
    """

    target_res: Union[None, str, int, List[int]] = None
    """
    Target resolution of the generated images and corresponding label maps.
    If None, the outputs will have the same resolution as the input label maps.
    Can be a number (isotropic resolution), or the path to a 1d numpy array.
    """

    output_shape: Union[None, str, int, List[int]] = None
    """
    Desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.
    """

    # GMM-sampling parameters

    generation_classes: Union[None, str, List[int]] = None
    """
    Indices regrouping generation labels into classes of same intensity
    distribution. Regrouped labels will thus share the same Gaussian when sampling a new image. Should be the path to a
    1d numpy array with the same length as generation_labels. and contain values between 0 and K-1, where K is the total
    number of classes. Default is all labels have different classes.
    Can be a list or a 1d numpy array, or the path to such an array.
    """

    prior_distributions: str = 'uniform'
    """
    Type of distribution from which we sample the GMM parameters.
    Can either be 'uniform', or 'normal'. Default is 'uniform'.
    """

    # Todo: Check how easy it is to serialize and deserialize nested integer lists
    prior_means: Union[None, str, List[List[float]]] = None
    """
    Hyperparameters controlling the prior distributions of the GMM means. Because
    these prior distributions are uniform or normal, they require by 2 hyperparameters. Can be a path to:
    1) an array of shape (2, K), where K is the number of classes (K=len(generation_labels) if generation_classes is
    not given). The mean of the Gaussian distribution associated to class k in [0, ...K-1] is sampled at each mini-batch
    from U(prior_means[0,k], prior_means[1,k]) if prior_distributions is uniform, and from
    N(prior_means[0,k], prior_means[1,k]) if prior_distributions is normal.
    2) an array of shape (2*n_mod, K), where each block of two rows is associated to hyperparameters derived
    from different modalities. In this case, if use_specific_stats_for_channel is False, we first randomly select a
    modality from the n_mod possibilities, and we sample the GMM means like in 2).
    If use_specific_stats_for_channel is True, each block of two rows correspond to a different channel
    (n_mod=n_channels), thus we select the corresponding block to each channel rather than randomly drawing it.
    Default is None, which corresponds all GMM means sampled from uniform distribution U(25, 225).
    """

    prior_stds: Union[None, str, List[List[float]]] = None
    """
    same as prior_means but for the standard deviations of the GMM.
    Default is None, which corresponds to U(5, 25).
    """

    use_specific_stats_for_channel: bool = False
    """
    Whether the i-th block of two rows in the prior arrays must be only used to generate the i-th channel.
    If True, n_mod should be equal to n_channels.
    """

    mix_prior_and_random: bool = False
    """
    If prior_means is not None, enables to reset the priors to their default
    values for half of these cases, and thus generate images of random contrast.
    """

    # Spatial deformation parameters

    flipping: bool = True
    """
    Iff True, introduce right/left random flipping.
    """

    scaling_bounds: Union[float, str, bool] = .2
    """
    if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) the path to a numpy array of shape (2, n_dims), in which case the scaling factor in dimension i is sampled from
    the uniform distribution of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    3) False, in which case scaling is completely turned off.
    """

    rotation_bounds: Union[float, str, bool] = 15.0
    """
    Similar to scaling_bounds but for the rotation angle, except that for case 1 the
    bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    """

    shearing_bounds: Union[float, str, bool] = .012
    """
    Similar to scaling_bounds but for the shearing parameter.
    """

    translation_bounds: Union[float, str, bool] = False
    """
    Similar to scaling_bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None).
    """

    nonlin_std: float = 4.
    """
    Standard deviation of the normal distribution from which we sample the first tensor for synthesising the 
    deformation field. Set to 0 to completely deactivate elastic deformation.
    """

    nonlin_scale: float = .04
    """
    Ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the elastic deformation field.
    """

    # Blurring/resampling parameters

    randomise_res: bool = True
    """
    Whether to mimic images that would have been 1) acquired at low resolution, and
    2) resampled to high resolution. The low resolution is uniformly resampled at each minibatch from [1mm, 9mm].
    In that process, the images generated by sampling the GMM are: 1) blurred at the sampled LR, 2) downsampled at LR,
    and 3) resampled at target_resolution.
    """

    max_res_iso: Optional[float] = 4.
    """
    If randomise_res is True, this enables to control the upper bound of the uniform
    distribution from which we sample the random resolution U(min_res, max_res_iso), where min_res is the resolution of
    the input label maps. Set to None to deactivate it, but if randomise_res is
    True, at least one of max_res_iso or max_res_aniso must be given.
    """

    max_res_aniso: Union[None, float, List[float]] = 8.
    """
    If randomise_res is True, this enables to downsample the input volumes to a random LR in
    only 1 (random) direction. This is done by randomly selecting a direction i in the range [0, n_dims-1], and sampling
    a value in the corresponding uniform distribution U(min_res[i], max_res_aniso[i]), where min_res is the resolution
    of the input label maps. Can be a number, a sequence, or a 1d numpy array. Set to None to deactivate it, but if
    randomise_res is True, at least one of max_res_iso or max_res_aniso must be given.
    """

    data_res: Union[None, int, List[int], str] = None
    """
    Specific acquisition resolution to mimic, as opposed to random resolution sampled when
    randomise_res is True. This triggers a blurring which mimics the acquisition resolution, but downsampling is
    optional (see param downsample). Default for data_res is None, where images are slightly blurred. If the generated
    images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence, a 1d numpy array, or
    the path to a 1d numpy array. In the multi-modal case, it should be given as a numpy array (or a path) of size
    (n_mod, n_dims), where each row is the acquisition resolution of the corresponding channel.
    """

    thickness: Union[None, int, List[int], str] = None
    """
    If data_res is provided, we can further specify the slice thickness of the low
    resolution images to mimic. Must be provided in the same format as data_res. Default thickness = data_res.
    """

    # Bias field parameters

    bias_field_std: float = .5
    """
    If strictly positive, this triggers the corruption of images with a bias field.
    The bias field is obtained by sampling a first small tensor from a normal distribution, resizing it to
    full size, and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the
    std dev of the normal distribution from which we sample the first tensor.
    Set to 0 to completely deactivate bias field corruption.
    """

    bias_scale: float = .025
    """
    If bias_field_std is not False, this designates the ratio between the size of
    the input label maps and the size of the first sampled tensor for synthesising the bias field.
    """

    return_gradients: bool = False
    """
    Whether to return the synthetic image or the magnitude of its spatial gradient (computed with Sobel kernels).
    """

    # UNet architecture parameters

    n_levels: int = 5
    """
    Number of level for the Unet
    """

    nb_conv_per_level: int = 2
    """
    Number of convolutional layers per level.
    """

    conv_size: int = 3
    """
    Size of the convolution kernels.
    """

    unet_feat_count: int = 24
    """
    Number of feature for the first layer of the UNet.
    """

    feat_multiplier: int = 2
    """
    Multiplicative factor to determine the number of features at each new level.
    If this is set to 1, we will keep the number of feature maps constant throughout the network.
    A value of 2 will double them (resp. half) after each max-pooling (resp. upsampling). 3 will triple them, etc.
    """

    activation: str = "elu"
    """
    Activation for all convolution layers except the last, which will use softmax regardless. Can be 'elu', 'relu'.
    """

    # Training parameters

    lr: float = 1e-4
    """
    Learning rate for the training
    """

    wl2_epochs: int = 1
    """
    Number of epochs for which the network (except the soft-max layer) is trained with L2 norm loss function.
    """

    dice_epochs: int = 50
    """
    Number of epochs with the soft Dice loss function.
    """

    steps_per_epoch: int = 10000
    """
    Number of steps per epoch. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    """

    checkpoint: Optional[str] = None
    """
    Path of an already saved model to load before starting the training.
    """

    wandb: bool = False
    """
    Add a WandB callback when training with the Dice loss function.
    """

    wandb_log_freq: Union[str, int] = "epoch"
    """
    if "epoch", logs metrics at the end of each epoch.
    If "batch", logs metrics at the end of each batch. If an integer, logs metrics at the end of that
    many batches.
    """

    # Only for training_with_tfrecords

    tfrecords_dir: Optional[str] = None
    """
    Path to the directory that contains the TFRecords. Only needed when training with TFRecords.
    """

    compression_type: str = ""
    """
    One of "GZIP", "ZLIB" or "" (no compression).
    Passed on to `tf.data.TFRecordDataset`:
    https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    """

    num_parallel_reads: Optional[int] = None
    """ 
    Number of files to read in parallel. If greater than one, the records of files read in 
    parallel are outputted in an interleaved order. If None, files will be read sequentially.
    Passed on to `tf.data.TFRecordDataset`:
    https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    """

    strategy: str = "null"
    """
    Specify the TF distributed strategy for the training. ONLY SUPPORTED WHEN TRAINING WITH TFRECORDS.
    Must be one of: 'null' (no distribution) or 'mirrored'.
    See https://www.tensorflow.org/guide/distributed_training for more information. 
    """

    use_original_unet: bool = True
    """
    Use the original implementation of the unet architecture? Otherwise we will use a custom implementation using more
    "standard" building blocks.
    The original implementation leads to a memory leak when trying to distribute the training over multiple GPUs.
    """

    def with_absolute_paths(self, reference_file: str):
        """
        Adds absolute paths to specified file paths in the TrainingOptions object.
        We just iterate through all properties and change the ones that are supposed to be paths.

        Args:
            reference_file (str): The reference file to be used for generating absolute paths.

        Returns:
            TrainingOptions: A copy of the TrainingOptions object with absolute paths added.
        """
        copy = TrainingOptions()
        non_path_properties = ["activation", "prior_distributions", "wandb_log_freq"]
        for key, value in vars(self).items():
            if isinstance(value, str) and key not in non_path_properties:
                setattr(copy, key, get_absolute_path(value, reference_file))
            else:
                setattr(copy, key, value)
        return copy
