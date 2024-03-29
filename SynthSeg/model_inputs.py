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
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def build_model_inputs(path_label_maps,
                       n_labels,
                       batchsize=1,
                       n_channels=1,
                       subjects_prob=None,
                       generation_classes=None,
                       prior_distributions='uniform',
                       prior_means=None,
                       prior_stds=None,
                       use_specific_stats_for_channel=False,
                       mix_prior_and_random=False):
    """
    This function builds a generator that will be used to give the necessary inputs to the label_to_image model: the
    input label maps, as well as the means and stds defining the parameters of the GMM (which change at each minibatch).
    :param path_label_maps: list of the paths of the input label maps.
    :param n_labels: number of labels in the input label maps.
    :param batchsize: (optional) numbers of images to generate per mini-batch. Default is 1.
    :param n_channels: (optional) number of channels to be synthesised. Default is 1.
    :param subjects_prob: (optional) relative order of importance (doesn't have to be probabilistic), with which to pick
    the provided label maps at each minibatch. Must be a 1D numpy array, as long as path_label_maps.
    :param generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
    distribution. Regrouped labels will thus share the same Gaussian when sampling a new image. Can be a sequence or a
    1d numpy array. It should have the same length as generation_labels, and contain values between 0 and K-1, where K
    is the total number of classes. Default is all labels have different classes.
    :param prior_distributions: (optional) type of distribution from which we sample the GMM parameters.
    Can either be 'uniform', or 'normal'. Default is 'uniform'.
    :param prior_means: (optional) hyperparameters controlling the prior distributions of the GMM means. Because
    these prior distributions are uniform or normal, they require by 2 hyperparameters. Thus prior_means can be:
    1) a sequence of length 2, directly defining the two hyperparameters: [min, max] if prior_distributions is
    uniform, [mean, std] if the distribution is normal. The GMM means of are independently sampled at each
    mini_batch from the same distribution.
    2) an array of shape (2, K), where K is the number of classes (K=len(generation_labels) if generation_classes is
    not given). The mean of the Gaussian distribution associated to class k in [0, ...K-1] is sampled at each mini-batch
    from U(prior_means[0,k], prior_means[1,k]) if prior_distributions is uniform, or from
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
    """

    # allocate unique class to each label if generation classes is not given
    if generation_classes is None:
        generation_classes = np.arange(n_labels)
    n_classes = len(np.unique(generation_classes))

    # make sure subjects_prob sums to 1
    subjects_prob = utils.load_array_if_path(subjects_prob)
    if subjects_prob is not None:
        subjects_prob /= np.sum(subjects_prob)

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.choice(np.arange(len(path_label_maps)), size=batchsize, p=subjects_prob)

        # initialise input lists
        list_label_maps = []
        list_means = []
        list_stds = []

        for idx in indices:

            # load input label map
            lab = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            if (npr.uniform() > 0.7) & ('seg_cerebral' in path_label_maps[idx]):
                lab[lab == 24] = 0

            # add label map to inputs
            list_label_maps.append(utils.add_axis(lab, axis=[0, -1]))

            # add means and standard deviations to inputs
            means = np.empty((1, n_labels, 0))
            stds = np.empty((1, n_labels, 0))
            for channel in range(n_channels):

                # retrieve channel specific stats if necessary
                if isinstance(prior_means, np.ndarray):
                    if (prior_means.shape[0] > 2) & use_specific_stats_for_channel:
                        if prior_means.shape[0] / 2 != n_channels:
                            raise ValueError("the number of blocks in prior_means does not match n_channels. This "
                                             "message is printed because use_specific_stats_for_channel is True.")
                        tmp_prior_means = prior_means[2 * channel:2 * channel + 2, :]
                    else:
                        tmp_prior_means = prior_means
                else:
                    tmp_prior_means = prior_means
                if (prior_means is not None) & mix_prior_and_random & (npr.uniform() > 0.5):
                    tmp_prior_means = None
                if isinstance(prior_stds, np.ndarray):
                    if (prior_stds.shape[0] > 2) & use_specific_stats_for_channel:
                        if prior_stds.shape[0] / 2 != n_channels:
                            raise ValueError("the number of blocks in prior_stds does not match n_channels. This "
                                             "message is printed because use_specific_stats_for_channel is True.")
                        tmp_prior_stds = prior_stds[2 * channel:2 * channel + 2, :]
                    else:
                        tmp_prior_stds = prior_stds
                else:
                    tmp_prior_stds = prior_stds
                if (prior_stds is not None) & mix_prior_and_random & (npr.uniform() > 0.5):
                    tmp_prior_stds = None

                # draw means and std devs from priors
                tmp_classes_means = utils.draw_value_from_distribution(tmp_prior_means, n_classes, prior_distributions,
                                                                       125., 125., positive_only=True)
                tmp_classes_stds = utils.draw_value_from_distribution(tmp_prior_stds, n_classes, prior_distributions,
                                                                      15., 15., positive_only=True)
                random_coef = npr.uniform()
                if random_coef > 0.95:  # reset the background to 0 in 5% of cases
                    tmp_classes_means[0] = 0
                    tmp_classes_stds[0] = 0
                elif random_coef > 0.7:  # reset the background to low Gaussian in 25% of cases
                    tmp_classes_means[0] = npr.uniform(0, 15)
                    tmp_classes_stds[0] = npr.uniform(0, 5)
                tmp_means = utils.add_axis(tmp_classes_means[generation_classes], axis=[0, -1])
                tmp_stds = utils.add_axis(tmp_classes_stds[generation_classes], axis=[0, -1])
                means = np.concatenate([means, tmp_means], axis=-1)
                stds = np.concatenate([stds, tmp_stds], axis=-1)
            list_means.append(means)
            list_stds.append(stds)

        # build list of inputs for generation model
        list_inputs = [list_label_maps, list_means, list_stds]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
