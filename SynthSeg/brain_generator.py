# python imports
import numpy as np

# project imports
from .model_inputs import build_model_inputs
from .labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils, edit_volumes


class BrainGenerator:

    def __init__(self,
                 labels_dir,
                 generation_labels=None,
                 output_labels=None,
                 n_neutral_labels=None,
                 batchsize=1,
                 n_channels=1,
                 target_res=None,
                 output_shape=None,
                 output_div_by_n=None,
                 prior_distributions='uniform',
                 generation_classes=None,
                 prior_means=None,
                 prior_stds=None,
                 use_specific_stats_for_channel=False,
                 mix_prior_and_random=False,
                 flipping=True,
                 scaling_bounds=0.15,
                 rotation_bounds=15,
                 shearing_bounds=.012,
                 translation_bounds=False,
                 nonlin_std=3.,
                 nonlin_shape_factor=0.0625,
                 blur_background=True,
                 data_res=None,
                 thickness=None,
                 downsample=False,
                 blur_range=1.15,
                 crop_channel_2=None,
                 apply_bias_field=True,
                 bias_field_std=0.3,
                 bias_shape_factor=0.025):
        """
        This class is wrapper around the labels_to_image_model model. It contains the GPU model that generates images
        from labels maps, and a python generator that suplies the input data for this model.
        To generate pairs of image/labels you can just call the method generate_image() on an object of this class.

        :param labels_dir: path of folder with all input label maps, or to a single label map.

        # IMPORTANT !!!
        # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
        # these values refer to the RAS axes.

        # label maps-related parameters
        :param generation_labels: (optional) list of all possible label values in the input label maps.
        Default is None, where the label values are directly gotten from the provided label maps.
        If not None, can be a sequence or a 1d numpy array, or the path to a 1d numpy array.
        If flipping is true (i.e. right/left flipping is enabled), generation_labels should be organised as follows:
        background label first, then non-sided labels (e.g. CSF, brainstem, etc.), then all the structures of the same
        hemisphere (can be left or right), and finally all the corresponding contralateral structures in the same order.
        :param output_labels: (optional) list of all the label values to keep in the output label maps (in no particular
        order). Should be a subset of the values contained in generation_labels.
        Label values that are in generation_labels but not in output_labels are reset to zero.
        Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.
        By default output labels are equal to generation labels.
        :param n_neutral_labels: (optional) number of non-sided generation labels.
        Default is total number of label values.

        # output-related parameters
        :param batchsize: (optional) numbers of images to generate per mini-batch. Default is 1.
        :param n_channels: (optional) number of channels to be synthetised. Default is 1.
        :param target_res: (optional) target resolution of the generated images and corresponding label maps.
        If None, the outputs will have the same resolution as the input label maps.
        Can be a number (isotropic resolution), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param output_shape: (optional) shape of the output image, obtained by randomly cropping the generated image.
        Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        Default is None, where no cropping is performed.
        :param output_div_by_n: (optional) forces the output shape to be divisible by this value. It overwrites
        output_shape if necessary. Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or
        the path to a 1d numpy array.

        # GMM-sampling parameters
        :param generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
        distribution. Regouped labels will thus share the same Gaussian when samling a new image. Can be a sequence, a
        1d numpy array, or the path to a 1d numpy array. It should have the same length as generation_labels, and
        contain values between 0 and K-1, where K is the total number of classes.
        Default is all labels have different classes (K=len(generation_labels)).
        :param prior_distributions: (optional) type of distribution from which we sample the GMM parameters.
        Can either be 'uniform', or 'normal'. Default is 'uniform'.
        :param prior_means: (optional) hyperparameters controlling the prior distributions of the GMM means. Because
        these prior distributions are uniform or normal, they require by 2 hyperparameters. Thus prior_means can be:
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
        :param prior_stds: (optional) same as prior_means but for the standard deviations of the GMM.
        Default is None, which corresponds to prior_stds = [5, 25].
        :param use_specific_stats_for_channel: (optional) whether the i-th block of two rows in the prior arrays must be
        only used to generate the i-th channel. If True, n_mod should be equal to n_channels. Default is False.
        :param mix_prior_and_random: (optional) if prior_means is not None, enables to reset the priors to their default
        values for half of thes cases, and thus generate images of random contrast.

        # spatial deformation parameters
        :param flipping: (optional) whether to introduce right/left random flipping. Default is True.
        :param scaling_bounds: (optional) range of the random saling to apply at each mini-batch. The scaling factor for
        each dimension is sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        [1-scaling_bounds, 1+scaling_bounds] for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) False, in which case scaling is completely turned off.
        Default is scaling_bounds = 0.15 (case 1)
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
        Default is rotation_bounds = 15.
        :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
        :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
        encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
        :param nonlin_std: (optional) Maximum value for the standard deviation of the normal distribution from which we
        sample the first tensor for synthesising the deformation field. Set to False if you wish to completely turn the
        elastic deformation off.
        :param nonlin_shape_factor: (optional) if nonlin_std is not False, factor between the shapes of the input label
        maps and the shape of the input non-linear tensor.

        # blurring/resampling parameters
        :param blur_background: (optional) whether to produce an unrealistic background or not.
        If True, the background is generated/blurred with the other labels, according to the values of prior_means and
        prior_stds. Also, it is reset to zero-background with a probability of 0.2.
        If False, the background is reset to zero, or can be replaced by a low-intensity background with a probability
        of 0.5. Additionally we correct for edge blurring effects.
        Default is True.
        :param data_res: (optional) acquisition resolution to mimick. If provided, the images sampled from the GMM are
        blurred to mimick data that would be: 1) acquired at the given acquisition resolution, and 2) resample at
        target_resolution.
        Default is None, where images are isotropically blurred to introduce some spatial correlation between voxels.
        If the generated images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence,
        a 1d numpy array, or the path to a 1d numy array. In the multi-modal case, it should be given as a numpy array
        (or a path) of size (n_mod, n_dims), where each row is the acquisition resolution of the correspionding chanel.
        :param thickness: (optional) if data_res is provided, we can further specify the slice thickness of the low
        resolution images to mimick.
        If the generated images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence,
        a 1d numpy array, or the path to a 1d numy array. In the multi-modal case, it should be given as a numpy array
        (or a path) of size (n_mod, n_dims), where each row is the acquisition resolution of the correspionding chanel.
        :param downsample: (optional) whether to actually downsample the volume image to data_res.
        Default is False, except when thickness is provided, and thickness < data_res.
        :param blur_range: (optional) Randomise the standard deviation of the blurring kernels, (whether data_res is
        given or not). At each mini_batch, the standard deviation of the blurring kernels are multiplied by a
        coefficient sampled from a uniform distribution with bounds [1/blur_range, blur_range].
        If None, no randomisation. Default is 1.15.
        :param crop_channel_2: (optional) stats for cropping second channel along the anterior-posterior axis.
        Should be a vector of length 4, with bounds of uniform distribution for cropping the front and back of the image
        (in percentage). None is no croppping.

        # bias field parameters
        :param apply_bias_field: (optional) whether to apply a bias field to the final image. Default is True.
        If True, the bias field is obtained by sampling a first tensor from normal distribution, resizing it to image
        size, and rescaling the values to positive number by taking the voxel-wise exponential. Default is True.
        :param bias_field_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the bias field.
        :param bias_shape_factor: (optional) If apply_bias_field is True, ratio between the size of the input
        label maps and the size of the sampled tensor for synthesising the bias field.
        """

        # prepare data files
        self.labels_paths = utils.list_images_in_folder(labels_dir)

        # generation parameters
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0], aff_ref=np.eye(4))
        self.n_channels = n_channels
        if generation_labels is not None:
            self.generation_labels = utils.load_array_if_path(generation_labels)
        else:
            self.generation_labels = utils.get_list_labels(labels_dir=labels_dir)
        if output_labels is not None:
            self.output_labels = utils.load_array_if_path(output_labels)
        else:
            self.output_labels = self.generation_labels
        if n_neutral_labels is not None:
            self.n_neutral_labels = n_neutral_labels
        else:
            self.n_neutral_labels = self.generation_labels.shape[0]
        self.target_res = utils.load_array_if_path(target_res)
        self.batchsize = batchsize
        # preliminary operations
        self.flipping = flipping
        self.output_shape = utils.load_array_if_path(output_shape)
        self.output_div_by_n = output_div_by_n
        # GMM parameters
        self.prior_distributions = prior_distributions
        if generation_classes is not None:
            self.generation_classes = utils.load_array_if_path(generation_classes)
            assert self.generation_classes.shape == self.generation_labels.shape, \
                'if provided, generation labels should have the same shape as generation_labels'
            unique_classes = np.unique(self.generation_classes)
            assert np.array_equal(unique_classes, np.arange(np.max(unique_classes)+1)), \
                'generation_classes should a linear range between 0 and its maximum value.'
        else:
            self.generation_classes = np.arange(self.generation_labels.shape[0])
        self.prior_means = utils.load_array_if_path(prior_means)
        self.prior_stds = utils.load_array_if_path(prior_stds)
        self.use_specific_stats_for_channel = use_specific_stats_for_channel
        # linear transformation parameters
        self.scaling_bounds = utils.load_array_if_path(scaling_bounds)
        self.rotation_bounds = utils.load_array_if_path(rotation_bounds)
        self.shearing_bounds = utils.load_array_if_path(shearing_bounds)
        self.translation_bounds = utils.load_array_if_path(translation_bounds)
        # elastic transformation parameters
        self.nonlin_std = nonlin_std
        self.nonlin_shape_factor = nonlin_shape_factor
        # blurring parameters
        self.blur_background = blur_background
        self.data_res = utils.load_array_if_path(data_res)
        self.thickness = utils.load_array_if_path(thickness)
        self.downsample = downsample
        self.blur_range = blur_range
        self.crop_second_channel = utils.load_array_if_path(crop_channel_2)
        # bias field parameters
        self.apply_bias_field = apply_bias_field
        self.bias_field_std = bias_field_std
        self.bias_shape_factor = bias_shape_factor

        # build transformation model
        self.labels_to_image_model, self.model_output_shape = self._build_labels_to_image_model()

        # build generator for model inputs
        self.model_inputs_generator = self._build_model_inputs_generator(mix_prior_and_random)

        # build brain generator
        self.brain_generator = self._build_brain_generator()

    def _build_labels_to_image_model(self):
        # build_model
        lab_to_im_model = labels_to_image_model(labels_shape=self.labels_shape,
                                                n_channels=self.n_channels,
                                                generation_labels=self.generation_labels,
                                                output_labels=self.output_labels,
                                                n_neutral_labels=self.n_neutral_labels,
                                                atlas_res=self.atlas_res,
                                                target_res=self.target_res,
                                                output_shape=self.output_shape,
                                                output_div_by_n=self.output_div_by_n,
                                                flipping=self.flipping,
                                                aff=np.eye(4),
                                                scaling_bounds=self.scaling_bounds,
                                                rotation_bounds=self.rotation_bounds,
                                                shearing_bounds=self.shearing_bounds,
                                                translation_bounds=self.translation_bounds,
                                                nonlin_std=self.nonlin_std,
                                                nonlin_shape_factor=self.nonlin_shape_factor,
                                                blur_background=self.blur_background,
                                                data_res=self.data_res,
                                                thickness=self.thickness,
                                                downsample=self.downsample,
                                                blur_range=self.blur_range,
                                                crop_channel2=self.crop_second_channel,
                                                apply_bias_field=self.apply_bias_field,
                                                bias_field_std=self.bias_field_std,
                                                bias_shape_factor=self.bias_shape_factor)
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]
        return lab_to_im_model, out_shape

    def _build_model_inputs_generator(self, mix_prior_and_random):
        # build model's inputs generator
        model_inputs_generator = build_model_inputs(path_label_maps=self.labels_paths,
                                                    n_labels=len(self.generation_labels),
                                                    batchsize=self.batchsize,
                                                    n_channels=self.n_channels,
                                                    generation_classes=self.generation_classes,
                                                    prior_means=self.prior_means,
                                                    prior_stds=self.prior_stds,
                                                    prior_distributions=self.prior_distributions,
                                                    use_specific_stats_for_channel=self.use_specific_stats_for_channel,
                                                    mix_prior_and_random=mix_prior_and_random)
        return model_inputs_generator

    def _build_brain_generator(self):
        while True:
            model_inputs = next(self.model_inputs_generator)
            [image, labels] = self.labels_to_image_model.predict(model_inputs)
            yield image, labels

    def generate_brain(self):
        """call this method when an object of this class has been instantiated to generate new brains"""
        (image, labels) = next(self.brain_generator)
        # put back images in native space
        list_images = list()
        list_labels = list()
        for i in range(self.batchsize):
            list_images.append(edit_volumes.align_volume_to_ref(image[i], np.eye(4),
                                                                aff_ref=self.aff, n_dims=self.n_dims))
            list_labels.append(edit_volumes.align_volume_to_ref(labels[i], np.eye(4),
                                                                aff_ref=self.aff, n_dims=self.n_dims))
        image = np.stack(list_images, axis=0)
        labels = np.stack(list_labels, axis=0)
        return image, labels
