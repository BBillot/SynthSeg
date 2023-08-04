from __future__ import annotations
from simple_parsing.helpers import Serializable
from dataclasses import dataclass

from .option_types import *
from .option_utils import get_absolute_path


@dataclass
class GeneratorOptions(Serializable):
    """
    Options for synthesizing brain MRI images from label maps
    """

    labels_dir: str = "../../data/training_label_maps"
    """Path of folder with all input label maps, or to a single label map."""

    generation_labels: Union[None, List[int], str] = None
    """
    List of all possible label values in the input label maps.
    Default is None, where the label values are directly gotten from the provided label maps.
    If not None, can be a sequence or a 1d numpy array, or the path to a 1d numpy array.
    If flipping is true (i.e. right/left flipping is enabled), generation_labels should be organised as follows:
    background label first, then non-sided labels (e.g. CSF, brainstem, etc.), then all the structures of the same
    hemisphere (can be left or right), and finally all the corresponding contra-lateral structures in the same order.
    """

    n_neutral_labels: Optional[int] = None
    """
    Number of non-sided generation labels. This is important only if you use
    flipping augmentation. Default is total number of label values.
    """

    output_labels: Union[None, List[int], str] = None
    """
    List of the same length as generation_labels to indicate which values to use in
    the label maps returned by this function, i.e. all occurrences of generation_labels[i] in the input label maps
    will be converted to output_labels[i] in the returned label maps. Examples:
    Set output_labels[i] to zero if you wish to erase the value generation_labels[i] from the returned label maps.
    Set output_labels[i]=generation_labels[i] to keep the value generation_labels[i] in the returned maps.
    Can be a list or a 1d numpy array. By default output_labels is equal to generation_labels.
    """

    subjects_prob: Union[None, List[int], str] = None
    """
    Relative order of importance (doesn't have to be probabilistic), with which to
    pick the provided label maps at each minibatch. Can be a sequence, a 1D numpy array, or the path to such an
    array, and it must be as long as path_label_maps.
    By default, all label maps are chosen with the same importance.
    """

    batchsize: int = 1
    """Numbers of images to generate per mini-batch."""

    n_channels: int = 1
    """Number of channels to be synthesised."""

    target_res: Union[None, int, List[int], str] = None
    """
    Target resolution of the generated images and corresponding label maps.
    If None, the outputs will have the same resolution as the input label maps.
    Can be a number (isotropic resolution), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    """

    output_shape: Union[None, int, List[int], str] = None
    """
    Shape of the output image, obtained by randomly cropping the generated image.
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.
    """

    output_div_by_n: Union[None, int, List[int], str] = None
    """
    Forces the output shape to be divisible by this value. It overwrites
    output_shape if necessary. Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or
    the path to a 1d numpy array.
    """

    # GMM-sampling parameters

    generation_classes: Union[None, List[int], str] = None
    """
    Indices regrouping generation labels into classes of same intensity
    distribution. Regrouped labels will thus share the same Gaussian when sampling a new image. Can be a sequence, a
    1d numpy array, or the path to a 1d numpy array. It should have the same length as generation_labels, and
    contain values between 0 and K-1, where K is the total number of classes.
    Default is all labels have different classes (K=len(generation_labels)).
    """

    prior_distributions: str = 'uniform'
    """
    type of distribution from which we sample the GMM parameters.
    Can either be 'uniform', or 'normal'. Default is 'uniform'.
    """

    prior_means: Union[None, List[int], List[List[int]]] = None
    """
    Controls the prior distributions of the GMM means. Because
    these prior distributions are uniform or normal, they require 2 value. Thus prior_means can be:
    1) a sequence of length 2, directly defining the two hyperparameters: [min, max] if prior_distributions is
    uniform, [mean, std] if the distribution is normal. The GMM means of are independently sampled at each
    mini_batch from the same distribution.
    2) an array of shape (2, K), where K is the number of classes (K=len(generation_labels) if generation_classes is
    not given). The mean of the Gaussian distribution associated to class k in [0, ...K-1] is sampled at each
    mini-batch from U(prior_means[0,k], prior_means[1,k]) if prior_distributions is uniform, and from
    N(prior_means[0,k], prior_means[1,k]) if prior_distributions is normal.
    3) an array of shape (2*n_mod, K), where each block of two rows is associated to hyperparameters derived
    from different modalities. In this case, if use_specific_stats_for_channel is False, we first randomly select a
    modality from the n_mod possibilities, and we sample the GMM means like in 2).
    If use_specific_stats_for_channel is True, each block of two rows correspond to a different channel
    (n_mod=n_channels), thus we select the corresponding block to each channel rather than randomly drawing it.
    4) the path to such a numpy array.
    Default is None, which corresponds to prior_means = [25, 225].
    """

    prior_stds: Union[None, List[int]] = None
    """
    Same as prior_means but for the standard deviations of the GMM.
    Default is None, which corresponds to prior_stds = [5, 25].
    """

    use_specific_stats_for_channel: bool = False
    """
    Whether the i-th block of two rows in the prior arrays must be
    only used to generate the i-th channel. If True, n_mod should be equal to n_channels. Default is False.
    """

    mix_prior_and_random: bool = False
    """
    If prior_means is not None, enables to reset the priors to their default
    values for half of these cases, and thus generate images of random contrast.
    """

    # spatial deformation parameters

    flipping: bool = True
    """
    Whether to introduce right/left random flipping.
    """

    scaling_bounds: Union[bool, float, List[float]] = .2
    """
    range of the random sampling to apply at each mini-batch. The scaling factor
    for each dimension is sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    [1-scaling_bounds, 1+scaling_bounds] for each dimension.
    2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
    (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
    3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
     of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    4) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.2 (case 1)
    """

    rotation_bounds: Union[int, List[int], bool] = 15
    """
    Same as scaling bounds but for the rotation angle, except that for cases 1
    and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
    """

    shearing_bounds: Union[bool, float, List[float]] = .012
    """
    Same as scaling bounds.
    """

    translation_bounds: Union[bool, float, List[float]] = False
    """
    same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
    """

    nonlin_std: float = 3.
    """
    Maximum value for the standard deviation of the normal distribution from which we
    sample the first tensor for synthesising the deformation field. Set to 0 if you wish to completely turn the
    elastic deformation off.
    """

    nonlin_scale: float = .04
    """
    if nonlin_std is strictly positive, factor between the shapes of the
    input label maps and the shape of the input non-linear tensor.
    """

    # blurring/resampling parameters

    randomise_res: bool = True
    """
    whether to mimic images that would have been 1) acquired at low resolution, and
    2) resampled to high resolution. The low resolution is uniformly resampled at each minibatch from [1mm, 9mm].
    In that process, the images generated by sampling the GMM are:
    1) blurred at the sampled LR, 2) down-sampled at LR, and 3) resampled at target_resolution.
    """

    max_res_iso: Optional[float] = 4.
    """
    If randomise_res is True, this enables to control the upper bound of the uniform
    distribution from which we sample the random resolution U(min_res, max_res_iso), where min_res is the resolution
    of the input label maps. Must be a number, and default is 4. Set to None to deactivate it, but if randomise_res
    is True, at least one of max_res_iso or max_res_aniso must be given.
    """

    max_res_aniso: Optional[float] = 8.
    """
    If randomise_res is True, this enables to down-sample the input volumes to a random LR
    in only 1 (random) direction. This is done by randomly selecting a direction i in range [0, n_dims-1], and
    sampling a value in the corresponding uniform distribution U(min_res[i], max_res_aniso[i]), where min_res is the
    resolution of the input label maps. Can be a number, a sequence, or a 1d numpy array. Set to None to deactivate
    it, but if randomise_res is True, at least one of max_res_iso or max_res_aniso must be given.

    """

    data_res: Union[None, int, List[int], str] = None
    """
    Specific acquisition resolution to mimic, as opposed to random resolution sampled
    when randomise_res is True. This triggers a blurring which mimics the acquisition resolution, but down-sampling
    is optional (see param down-sample). Default for data_res is None, where images are slightly blurred.
    If the generated images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence,
    a 1d numpy array, or the path to a 1d numpy array. In the multi-modal case, it should be given as a numpy array
    (or a path) of size (n_mod, n_dims), where each row is the acquisition resolution of the corresponding channel.
    """

    thickness: Union[None, int, List[int], str] = None
    """
    if data_res is provided, we can further specify the slice thickness of the low
    resolution images to mimic. Must be provided in the same format as data_res. Default thickness = data_res.
    """

    # bias field parameters

    bias_field_std: float = .5
    """
    If strictly positive, this triggers the corruption of synthesised images with
    a bias field. It is obtained by sampling a first small tensor from a normal distribution, resizing it to full
    size, and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the
    std dev of the normal distribution from which we sample the first tensor. Set to 0 to deactivate bias field.
    """

    bias_scale: float = .025
    """
    If bias_field_std is strictly positive, this designates the ratio between
    the size of the input label maps and the size of the first sampled tensor for synthesising the bias field.
    """

    return_gradients: bool = False
    """
    whether to return the synthetic image or the magnitude of its spatial
    gradient (computed with Sobel kernels).
    """

    def with_absolute_paths(self, reference_file: str):
        """
        Adds absolute paths to specified file paths in the GeneratorOptions object.
        Since all string properties are supposed to be paths, we just iterate through all properties
        and change the ones that are strings.

        Args:
            reference_file (str): The reference file to be used for generating absolute paths.

        Returns:
            GeneratorOptions: A copy of the GeneratorOptions object with absolute paths added.
        """
        copy = GeneratorOptions()
        excluded_properties = ["prior_distributions"]
        for key, value in vars(self).items():
            if isinstance(value, str) and key not in excluded_properties:
                setattr(copy, key, get_absolute_path(value, reference_file))
            else:
                setattr(copy, key, value)
        return copy
