# python imports
import numpy as np
import numpy.random as npr

# project imports
from . import utils
from .lab2im_model import lab2im_model


class ImageGenerator:

    def __init__(self,
                 labels_dir,
                 generation_labels=None,
                 output_labels=None,
                 batch_size=1,
                 n_channels=1,
                 target_res=None,
                 output_shape=None,
                 output_div_by_n=None,
                 generation_classes=None,
                 prior_distributions='uniform',
                 prior_means=None,
                 prior_stds=None,
                 use_specific_stats_for_channel=False,
                 blur_background=True,
                 blur_range=1.15):
        """
        This class is wrapper around the lab2im_model model. It contains the GPU model that generates images from labels
        maps, and a python generator that suplies the input data for this model.
        To generate pairs of image/labels you can just call the method generate_image() on an object of this class.

        :param labels_dir: path of folder with all input label maps, or to a single label map.

        # IMPORTANT !!!
        # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
        # these values refer to the axes of the raw label map (i.e. once it has been loaded in python).
        # Depending on the label map orientation, the axes of its raw array may or may not correspond to the RAS axes.

        # label maps-related parameters
        :param generation_labels: (optional) list of all possible label values in the input label maps.
        Default is None, where the label values are directly gotten from the provided label maps.
        If not None, can be a sequence or a 1d numpy array, or the path to a 1d numpy array.
        :param output_labels: (optional) list of all the label values to keep in the output label maps.
        Should be a subset of the values contained in generation_labels.
        Label values that are in generation_labels but not in output_labels are reset to zero.
        Can be a sequence, a 1d numpy array, or the path to a 1d numpy array.

        # output-related parameters
        :param batch_size: (optional) numbers of images to generate per mini-batch. Default is 1.
        :param n_channels: (optional) number of channels to be synthetised. Default is 1.
        :param target_res: (optional) target resolution of the generated images and corresponding label maps.
        If None, the outputs will have the same resolution as the input label maps.
        Can be a number (isotropic resolution), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param output_shape: (optional) shape of the output image, obtained by randomly cropping the generated image.
        Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
        :param output_div_by_n: (optional) forces the output shape to be divisible by this value. It overwrites
        output_shape if necessary. Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or
        the path to a 1d numpy array.

        # GMM-sampling parameters
        :param generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
        distribution. Regouped labels will thus share the same Gaussian when samling a new image. Can be a sequence, a
        1d numpy array, or the path to a 1d numpy array.
        It should have the same length as generation_labels, and contain values between 0 and K-1, where K is the total
        number of classes. Default is all labels have different classes (K=len(generation_labels)).
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

        # blurring parameters
        :param blur_background: (optional) If True, the background is blurred with the other labels, and can be reset to
        zero with a probability of 0.2. If False, the background is not blurred (we apply an edge blurring correction),
        and can be replaced by a low-intensity background with a probability of 0.5.
        :param blur_range: (optional) Randomise the standard deviation of the blurring kernels, (whether data_res is
        given or not). At each mini_batch, the standard deviation of the blurring kernels are multiplied by a c
        oefficient sampled from a uniform distribution with bounds [1/blur_range, blur_range].
        If None, no randomisation. Default is 1.15.
        """

        # prepare data files
        if ('.nii.gz' in labels_dir) | ('.nii' in labels_dir) | ('.mgz' in labels_dir) | ('.npz' in labels_dir):
            self.labels_paths = [labels_dir]
        else:
            self.labels_paths = utils.list_images_in_folder(labels_dir)
        assert len(self.labels_paths) > 0, "Could not find any training data"

        # generation parameters
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0])
        self.n_channels = n_channels
        if generation_labels is not None:
            self.generation_labels = utils.load_array_if_path(generation_labels)
        else:
            self.generation_labels = utils.get_list_labels(labels_dir=labels_dir)
        if output_labels is not None:
            self.output_labels = utils.load_array_if_path(output_labels)
        else:
            self.output_labels = self.generation_labels
        self.target_res = utils.load_array_if_path(target_res)
        # preliminary operations
        self.output_shape = utils.load_array_if_path(output_shape)
        self.output_div_by_n = output_div_by_n
        # GMM parameters
        if generation_classes is not None:
            self.generation_classes = utils.load_array_if_path(generation_classes)
            assert self.generation_classes.shape == self.generation_labels.shape, \
                'if provided, generation labels should have the same shape as generation_labels'
            unique_classes = np.unique(self.generation_classes)
            assert np.array_equal(unique_classes, np.arange(np.max(unique_classes)+1)), \
                'generation_classes should a linear range between 0 and its maximum value.'
        else:
            self.generation_classes = np.arange(self.generation_labels.shape[0])
        self.prior_distributions = prior_distributions
        self.prior_means = utils.load_array_if_path(prior_means)
        self.prior_stds = utils.load_array_if_path(prior_stds)
        self.use_specific_stats_for_channel = use_specific_stats_for_channel

        # blurring parameters
        self.blur_background = blur_background
        self.blur_range = blur_range

        # build transformation model
        self.labels_to_image_model, self.model_output_shape = self._build_lab2im_model()

        # build generator for model inputs
        self.model_inputs_generator = self._build_model_inputs(len(self.generation_labels), batch_size)

        # build brain generator
        self.image_generator = self._build_image_generator()

    def _build_lab2im_model(self):
        # build_model
        lab_to_im_model = lab2im_model(labels_shape=self.labels_shape,
                                       n_channels=self.n_channels,
                                       generation_labels=self.generation_labels,
                                       output_labels=self.output_labels,
                                       atlas_res=self.atlas_res,
                                       target_res=self.target_res,
                                       output_shape=self.output_shape,
                                       output_div_by_n=self.output_div_by_n,
                                       blur_background=self.blur_background,
                                       blur_range=self.blur_range)
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]
        return lab_to_im_model, out_shape

    def _build_image_generator(self):
        while True:
            model_inputs = next(self.model_inputs_generator)
            [image, labels] = self.labels_to_image_model.predict(model_inputs)
            yield image, labels

    def generate_image(self):
        """call this method when an object of this class has been instantiated to generate new brains"""
        (image, labels) = next(self.image_generator)
        return np.squeeze(image), np.squeeze(labels)

    def _build_model_inputs(self, n_labels, batch_size=1):

        # get label info
        labels_shape, _, n_dims, _, _, _ = utils.get_volume_info(self.labels_paths[0])

        # Generate!
        while True:

            # randomly pick as many images as batch_size
            label_map_indices = npr.randint(len(self.labels_paths), size=batch_size)

            # initialise input tensors
            y_all = []
            means_all = []
            std_devs_all = []
            aff_all = []

            for label_map_idx in label_map_indices:

                # add labels to inputs
                y = utils.load_volume(self.labels_paths[label_map_idx], dtype='int')
                y_all.append(utils.add_axis(y, axis=-2))

                # add means and standard deviations to inputs
                means = np.empty((n_labels, 0))
                std_devs = np.empty((n_labels, 0))
                for channel in range(self.n_channels):

                    # retrieve channel specific stats if necessary
                    if isinstance(self.prior_means, np.ndarray):
                        if self.prior_means.shape[0] > 2 & self.use_specific_stats_for_channel:
                            if self.prior_means.shape[0] / 2 != self.n_channels:
                                raise ValueError("the number of blocks in prior_means does not match n_channels. This "
                                                 "message is printed because use_specific_stats_for_channel is True.")
                            tmp_prior_means = self.prior_means[2 * channel:2 * channel + 2, :]
                        else:
                            tmp_prior_means = self.prior_means
                    else:
                        tmp_prior_means = self.prior_means
                    if isinstance(self.prior_stds, np.ndarray):
                        if self.prior_stds.shape[0] > 2 & self.use_specific_stats_for_channel:
                            if self.prior_stds.shape[0] / 2 != self.n_channels:
                                raise ValueError("the number of blocks in prior_stds does not match n_channels. This "
                                                 "message is printed because use_specific_stats_for_channel is True.")
                            tmp_prior_stds = self.prior_stds[2 * channel:2 * channel + 2, :]
                        else:
                            tmp_prior_stds = self.prior_stds
                    else:
                        tmp_prior_stds = self.prior_stds

                    # draw means and std devs from priors
                    tmp_classes_means = utils.draw_value_from_distribution(tmp_prior_means, n_labels,
                                                                           self.prior_distributions, 125., 100.,
                                                                           positive_only=True)
                    tmp_classes_stds = utils.draw_value_from_distribution(tmp_prior_stds, n_labels,
                                                                          self.prior_distributions, 15., 10.,
                                                                          positive_only=True)
                    tmp_means = utils.add_axis(tmp_classes_means[self.generation_classes], -1)
                    tmp_stds = utils.add_axis(tmp_classes_stds[self.generation_classes], -1)
                    means = np.concatenate([means, tmp_means], axis=1)
                    std_devs = np.concatenate([std_devs, tmp_stds], axis=1)
                means_all.append(utils.add_axis(means))
                std_devs_all.append(utils.add_axis(std_devs))

                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = utils.draw_value_from_distribution(None, size=n_dims, centre=1, default_range=.07)
                if n_dims == 2:
                    rotation = utils.draw_value_from_distribution(None, default_range=10.0)
                else:
                    rotation = utils.draw_value_from_distribution(None, size=n_dims, default_range=10.0)
                shearing = utils.draw_value_from_distribution(None, size=n_dims ** 2 - n_dims, default_range=.007)
                aff = utils.create_affine_transformation_matrix(n_dims, scaling, rotation, shearing)
                aff_all.append(utils.add_axis(aff))

            # build list of inputs to augmentation model
            inputs_vals = [y_all, means_all, std_devs_all, aff_all]

            # put images and labels (concatenated if batch_size>1) into a tuple of 2 elements: (cat_images, cat_labels)
            if batch_size > 1:
                inputs_vals = [np.concatenate(item, 0) for item in inputs_vals]
            else:
                inputs_vals = [item[0] for item in inputs_vals]

            yield inputs_vals
