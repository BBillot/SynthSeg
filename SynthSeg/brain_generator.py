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
import numpy as np

# project imports
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg.labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils, edit_volumes


class BrainGenerator:

    def __init__(self,
                 labels_dir,
                 generation_labels=None,
                 n_neutral_labels=None,
                 output_labels=None,
                 subjects_prob=None,
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
                 scaling_bounds=.2,
                 rotation_bounds=15,
                 shearing_bounds=.012,
                 translation_bounds=False,
                 nonlin_std=4.,
                 nonlin_scale=.04,
                 randomise_res=True,
                 max_res_iso=4.,
                 max_res_aniso=8.,
                 data_res=None,
                 thickness=None,
                 bias_field_std=.7,
                 bias_scale=.025,
                 return_gradients=False):
        """
        This class is wrapper around the labels_to_image_model model. It contains the GPU model that generates images
        from labels maps, and a python generator that supplies the input data for this model.
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
        :param n_neutral_labels: (optional) number of non-sided generation labels. This is important only if you use
        flipping augmentation. Default is total number of label values.
        :param output_labels: (optional) list of the same length as generation_labels to indicate which values to use in
        the label maps returned by this function, i.e. all occurrences of generation_labels[i] in the input label maps
        will be converted to output_labels[i] in the returned label maps. Examples:
        Set output_labels[i] to zero if you wish to erase the value generation_labels[i] from the returned label maps.
        Set output_labels[i]=generation_labels[i] to keep the value generation_labels[i] in the returned maps.
        Can be a list or a 1d numpy array. By default output_labels is equal to generation_labels.
        :param subjects_prob: (optional) relative order of importance (doesn't have to be probabilistic), with which to
        pick the provided label maps at each minibatch. Can be a sequence, a 1D numpy array, or the path to such an
        array, and it must be as long as path_label_maps. By default, all label maps are chosen with the same importance

        # output-related parameters
        :param batchsize: (optional) numbers of images to generate per mini-batch. Default is 1.
        :param n_channels: (optional) number of channels to be synthesised. Default is 1.
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
        distribution. Regrouped labels will thus share the same Gaussian when sampling a new image. Can be a sequence, a
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
        values for half of these cases, and thus generate images of random contrast.

        # spatial deformation parameters
        :param flipping: (optional) whether to introduce right/left random flipping. Default is True.
        :param scaling_bounds: (optional) range of the random sampling to apply at each mini-batch. The scaling factor
        for each dimension is sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        [1-scaling_bounds, 1+scaling_bounds] for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) False, in which case scaling is completely turned off.
        Default is scaling_bounds = 0.2 (case 1)
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
        Default is rotation_bounds = 15.
        :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
        :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
        encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
        :param nonlin_std: (optional) Maximum value for the standard deviation of the normal distribution from which we
        sample the first tensor for synthesising the deformation field. Set to 0 if you wish to completely turn the
        elastic deformation off.
        :param nonlin_scale: (optional) if nonlin_std is strictly positive, factor between the shapes of the
        input label maps and the shape of the input non-linear tensor.

        # blurring/resampling parameters
        :param randomise_res: (optional) whether to mimic images that would have been 1) acquired at low resolution, and
        2) resampled to high resolution. The low resolution is uniformly resampled at each minibatch from [1mm, 9mm].
        In that process, the images generated by sampling the GMM are:
        1) blurred at the sampled LR, 2) downsampled at LR, and 3) resampled at target_resolution.
        :param max_res_iso: (optional) If randomise_res is True, this enables to control the upper bound of the uniform
        distribution from which we sample the random resolution U(min_res, max_res_iso), where min_res is the resolution
        of the input label maps. Must be a number, and default is 4. Set to None to deactivate it, but if randomise_res
        is True, at least one of max_res_iso or max_res_aniso must be given.
        :param max_res_aniso: If randomise_res is True, this enables to downsample the input volumes to a random LR
        in only 1 (random) direction. This is done by randomly selecting a direction i in range [0, n_dims-1], and
        sampling a value in the corresponding uniform distribution U(min_res[i], max_res_aniso[i]), where min_res is the
        resolution of the input label maps. Can be a number, a sequence, or a 1d numpy array. Set to None to deactivate
        it, but if randomise_res is True, at least one of max_res_iso or max_res_aniso must be given.
        :param data_res: (optional) specific acquisition resolution to mimic, as opposed to random resolution sampled
        when randomise_res is True. This triggers a blurring which mimics the acquisition resolution, but downsampling
        is optional (see param downsample). Default for data_res is None, where images are slightly blurred.
        If the generated images are uni-modal, data_res can be a number (isotropic acquisition resolution), a sequence,
        a 1d numpy array, or the path to a 1d numpy array. In the multi-modal case, it should be given as a numpy array
        (or a path) of size (n_mod, n_dims), where each row is the acquisition resolution of the corresponding channel.
        :param thickness: (optional) if data_res is provided, we can further specify the slice thickness of the low
        resolution images to mimic. Must be provided in the same format as data_res. Default thickness = data_res.

        # bias field parameters
        :param bias_field_std: (optional) If strictly positive, this triggers the corruption of synthesised images with
        a bias field. It is obtained by sampling a first small tensor from a normal distribution, resizing it to full
        size, and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the
        std dev of the normal distribution from which we sample the first tensor. Set to 0 to deactivate bias field.
        :param bias_scale: (optional) If bias_field_std is strictly positive, this designates the ratio between
        the size of the input label maps and the size of the first sampled tensor for synthesising the bias field.

        :param return_gradients: (optional) whether to return the synthetic image or the magnitude of its spatial
        gradient (computed with Sobel kernels).
        """

        # prepare data files
        self.labels_paths = utils.list_images_in_folder(labels_dir)
        if subjects_prob is not None:
            self.subjects_prob = np.array(utils.reformat_to_list(subjects_prob, load_as_numpy=True), dtype='float32')
            assert len(self.subjects_prob) == len(self.labels_paths), \
                'subjects_prob should have the same length as labels_path, ' \
                'had {} and {}'.format(len(self.subjects_prob), len(self.labels_paths))
        else:
            self.subjects_prob = None

        # generation parameters
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0], aff_ref=np.eye(4))
        self.n_channels = n_channels
        if generation_labels is not None:
            self.generation_labels = utils.load_array_if_path(generation_labels)
        else:
            self.generation_labels, _ = utils.get_list_labels(labels_dir=labels_dir)
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
                'if provided, generation_classes should have the same shape as generation_labels'
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
        self.nonlin_scale = nonlin_scale
        # blurring parameters
        self.randomise_res = randomise_res
        self.max_res_iso = max_res_iso
        self.max_res_aniso = max_res_aniso
        self.data_res = utils.load_array_if_path(data_res)
        assert not (self.randomise_res & (self.data_res is not None)), \
            'randomise_res and data_res cannot be provided at the same time'
        self.thickness = utils.load_array_if_path(thickness)
        # bias field parameters
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.return_gradients = return_gradients

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
                                                nonlin_scale=self.nonlin_scale,
                                                randomise_res=self.randomise_res,
                                                max_res_iso=self.max_res_iso,
                                                max_res_aniso=self.max_res_aniso,
                                                data_res=self.data_res,
                                                thickness=self.thickness,
                                                bias_field_std=self.bias_field_std,
                                                bias_scale=self.bias_scale,
                                                return_gradients=self.return_gradients)
        out_shape = lab_to_im_model.output[0].get_shape().as_list()[1:]
        return lab_to_im_model, out_shape

    def _build_model_inputs_generator(self, mix_prior_and_random):
        # build model's inputs generator
        model_inputs_generator = build_model_inputs(path_label_maps=self.labels_paths,
                                                    n_labels=len(self.generation_labels),
                                                    batchsize=self.batchsize,
                                                    n_channels=self.n_channels,
                                                    subjects_prob=self.subjects_prob,
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
        image = np.squeeze(np.stack(list_images, axis=0))
        labels = np.squeeze(np.stack(list_labels, axis=0))
        return image, labels
