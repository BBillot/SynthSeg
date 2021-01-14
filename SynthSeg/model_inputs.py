# python imports
import numpy as np
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def build_model_inputs(path_label_maps,
                       n_labels,
                       batchsize=1,
                       n_channels=1,
                       generation_classes=None,
                       prior_distributions='uniform',
                       prior_means=None,
                       prior_stds=None,
                       use_specific_stats_for_channel=False,
                       mix_prior_and_random=False,
                       apply_linear_trans=True,
                       scaling_bounds=0.15,
                       rotation_bounds=15,
                       shearing_bounds=.012,
                       background_paths=None):
    """
    This function builds a generator to be fed to the lab2im model. It enables to generate all the required inputs,
    according to the operations performed in the model.
    :param path_label_maps: list of the paths of the input label maps.
    :param n_labels: number of labels in the input label maps.
    :param batchsize: (optional) numbers of images to generate per mini-batch. Default is 1.
    :param n_channels: (optional) number of channels to be synthetised. Default is 1.
    :param generation_classes: (optional) Indices regrouping generation labels into classes of same intensity
    distribution. Regouped labels will thus share the same Gaussian when samling a new image. Can be a sequence or a
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
    values for half of thes cases, and thus generate images of random contrast.
    :param apply_linear_trans: (optional) whether to apply affine deformation. Default is True.
    :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
    (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
    3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
     of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    4) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
    and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    Default is rotation_bounds = 15.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param background_paths: (optional) list of paths of label maps to replace the soft brain tissues (label 258) with.
    """

    # get label info
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_label_maps[0])

    # allocate unique class to each label if generation classes is not given
    if generation_classes is None:
        generation_classes = np.arange(n_labels)

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_label_maps), size=batchsize)

        # initialise input lists
        list_label_maps = []
        list_means = []
        list_stds = []
        list_affine_transforms = []

        for idx in indices:

            # add labels to inputs
            lab = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(lab, axis=[0, -1]))

            # add means and standard deviations to inputs
            means = np.empty((n_labels, 0))
            stds = np.empty((n_labels, 0))
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
                tmp_classes_means = utils.draw_value_from_distribution(tmp_prior_means, n_labels, prior_distributions,
                                                                       125., 100., positive_only=True)
                tmp_classes_stds = utils.draw_value_from_distribution(tmp_prior_stds, n_labels, prior_distributions,
                                                                      15., 10., positive_only=True)
                tmp_means = utils.add_axis(tmp_classes_means[generation_classes], -1)
                tmp_stds = utils.add_axis(tmp_classes_stds[generation_classes], -1)
                means = np.concatenate([means, tmp_means], axis=1)
                stds = np.concatenate([stds, tmp_stds], axis=1)
            list_means.append(utils.add_axis(means))
            list_stds.append(utils.add_axis(stds))

            # add linear transform to inputs
            if apply_linear_trans:
                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = utils.draw_value_from_distribution(scaling_bounds, size=n_dims, centre=1, default_range=.15)
                if n_dims == 2:
                    rotation = utils.draw_value_from_distribution(rotation_bounds, default_range=15.0)
                else:
                    rotation = utils.draw_value_from_distribution(rotation_bounds, size=n_dims, default_range=15.0)
                shearing = utils.draw_value_from_distribution(shearing_bounds, size=n_dims**2-n_dims, default_range=.01)
                affine_transform = utils.create_affine_transformation_matrix(n_dims, scaling, rotation, shearing)
                list_affine_transforms.append(utils.add_axis(affine_transform))

        # build list of inputs of augmentation model
        list_inputs = [list_label_maps, list_means, list_stds]
        if apply_linear_trans:
            list_inputs.append(list_affine_transforms)

        # concatenate individual input types if batchsize > 1
        if batchsize > 1:
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs


def means_stds_fs_labels_with_relations(means_range, std_devs_range, min_diff=15, head=True):

    # draw gm wm and csf means
    gm_wm_csf_means = np.zeros(3)
    while (abs(gm_wm_csf_means[1] - gm_wm_csf_means[0]) < min_diff) | \
          (abs(gm_wm_csf_means[1] - gm_wm_csf_means[2]) < min_diff) | \
          (abs(gm_wm_csf_means[0] - gm_wm_csf_means[2]) < min_diff):
        gm_wm_csf_means = utils.draw_value_from_distribution(means_range, 3, 'uniform', 125., 100., positive_only=True)
        gm_wm_csf_means = utils.add_axis(gm_wm_csf_means, -1)

    # apply relations
    wm = gm_wm_csf_means[0]
    gm = gm_wm_csf_means[1]
    csf = gm_wm_csf_means[2]
    csf_like = csf * npr.uniform(low=0.95, high=1.05)
    alpha_thalamus = npr.uniform(low=0.4, high=0.9)
    thalamus = alpha_thalamus*gm + (1-alpha_thalamus)*wm
    cerebellum_wm = wm * npr.uniform(low=0.7, high=1.3)
    cerebellum_gm = gm * npr.uniform(low=0.7, high=1.3)
    caudate = gm * npr.uniform(low=0.9, high=1.1)
    putamen = gm * npr.uniform(low=0.9, high=1.1)
    hippocampus = gm * npr.uniform(low=0.9, high=1.1)
    amygdala = gm * npr.uniform(low=0.9, high=1.1)
    accumbens = caudate * npr.uniform(low=0.9, high=1.1)
    pallidum = wm * npr.uniform(low=0.8, high=1.2)
    brainstem = wm * npr.uniform(low=0.8, high=1.2)
    alpha_ventralDC = npr.uniform(low=0.1, high=0.6)
    ventralDC = alpha_ventralDC*gm + (1-alpha_ventralDC)*wm
    alpha_choroid = npr.uniform(low=0.0, high=1.0)
    choroid = alpha_choroid*csf + (1-alpha_choroid)*wm

    # regroup structures
    neutral_means = [np.zeros(1), csf_like, csf_like, brainstem, csf]
    sided_means = [wm, gm, csf_like, csf_like, cerebellum_wm, cerebellum_gm, thalamus, caudate, putamen, pallidum,
                   hippocampus, amygdala, accumbens, ventralDC, choroid]

    # draw std deviations
    std = utils.draw_value_from_distribution(std_devs_range, 17, 'uniform', 15., 10., positive_only=True)
    std = utils.add_axis(std, -1)
    neutral_stds = [np.zeros(1), std[1], std[1], std[2], std[3]]
    sided_stds = [std[4], std[5], std[1], std[1], std[6], std[7], std[8], std[9], std[10], std[11], std[12], std[13],
                  std[14], std[15], std[16]]

    # add means and variances for extra head labels if necessary
    if head:
        # means
        extra_means = utils.draw_value_from_distribution(means_range, 2, 'uniform', 125., 100., positive_only=True)
        extra_means = utils.add_axis(extra_means, -1)
        skull = extra_means[0]
        soft_non_brain = extra_means[1]
        eye = csf * npr.uniform(low=0.95, high=1.05)
        optic_chiasm = wm * npr.uniform(low=0.8, high=1.2)
        vessel = csf * npr.uniform(low=0.7, high=1.3)
        neutral_means += [csf_like, optic_chiasm, skull, soft_non_brain, eye]
        sided_means.insert(-1, vessel)
        # std dev
        extra_std = utils.draw_value_from_distribution(std_devs_range, 4, 'uniform', 15., 10., positive_only=True)
        extra_std = utils.add_axis(extra_std, -1)
        neutral_stds += [std[1], extra_std[0], extra_std[1], extra_std[2], std[1]]
        sided_stds.insert(-1, extra_std[3])

    means = np.concatenate([np.array(neutral_means), np.array(sided_means), np.array(sided_means)])
    stds = np.concatenate([np.array(neutral_stds), np.array(sided_stds), np.array(sided_stds)])

    return means, stds
