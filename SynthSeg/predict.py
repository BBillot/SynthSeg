# python imports
import os
import csv
import numpy as np
import keras.layers as KL
from keras.models import Model

# project imports
from SynthSeg import evaluate

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models


def predict(path_images,
            path_model,
            segmentation_label_list,
            dist_map=False,
            path_segmentations=None,
            path_posteriors=None,
            path_volumes=None,
            padding=None,
            cropping=None,
            resample=None,
            aff_ref='FS',
            sigma_smoothing=0,
            keep_biggest_component=False,
            conv_size=3,
            n_levels=5,
            nb_conv_per_level=2,
            unet_feat_count=24,
            feat_multiplier=2,
            activation='elu',
            gt_folder=None,
            evaluation_label_list=None,
            compute_distances=False,
            recompute=True,
            verbose=True):
    """
    This function uses trained models to segment images.
    It is crucial that the inputs match the architecture parameters of the trained model.
    :param path_images: path of the images to segment. Can be the path to a directory or the path to a single image.
    :param path_model: path ot the trained model.
    :param segmentation_label_list: List of labels for which to compute Dice scores. It should contain the same values
    as the segmentation label list used for training the network.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array.
    :param dist_map: (optional) whether the input will contain distance maps channels (between each intenisty channels)
    Default is False.
    :param path_segmentations: (optional) path where segmentations will be writen.
    Should be a dir, if path_images is a dir, and afile if path_images is a file.
    Should not be None, if path_posteriors is None.
    :param path_posteriors: (optional) path where posteriors will be writen.
    Should be a dir, if path_images is a dir, and afile if path_images is a file.
    Should not be None, if path_segmentations is None.
    :param path_volumes: (optional) path of a csv file where the soft volumes of all segmented regions will be writen.
    The rows of the csv file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    :param padding: (optional) pad the images to the specified shape before predicting the segmentation maps.
    Can be an int, a sequence or a 1d numpy array.
    :param cropping: (optional) crop the images to the specified shape before predicting the segmentation maps.
    If padding and cropping are specified, images are padded before being cropped.
    Can be an int, a sequence or a 1d numpy array.
    :param resample: (optional) resample the images to the specified resolution before predicting the segmentation maps.
    Can be an int, a sequence or a 1d numpy array.
    :param aff_ref: (optional) type of affine matrix of the images used for training. By default this is set to the
    FreeSurfer orientation ('FS'), as it was the configuration in which SynthSeg was trained. However, the new models
    are now trained on data aligned with identity vox2ras matrix, so you need to change aff_ref to 'identity'.
    :param sigma_smoothing: (optional) If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    :param keep_biggest_component: (optional) whether to only keep the biggest component in the predicted segmentation.
    :param conv_size: (optional) size of unet's convolution masks. Default is 3.
    :param n_levels: (optional) number of levels for unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolution layers per level. Default is 2.
    :param unet_feat_count: (optional) number of features for the first layer of the unet. Default is 24.
    :param feat_multiplier: (optional) multiplicative factor for the number of feature for each new level. Default is 2.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.
    :param gt_folder: (optional) folder containing ground truth files for evaluation.
    A numpy array containing all dice scores (labels in rows, subjects in columns) will be writen either at
    segmentations_dir (if not None), or posteriors_dir.
    :param evaluation_label_list: (optional) if gt_folder is True you can evaluate the Dice scores on a subset of the
    segmentation labels, by providing another label list here. Can be a sequence, a 1d numpy array, or the path to a
    numpy 1d array. Default is the same as segmentation_label_list.
    :param recompute: (optional) whether to recompute segmentations that were already computed. This also applies to
    Dice scores, if gt_folder is not None. Default is True.
    :param verbose: (optional) whether to print out info about the remaining number of cases.
    """

    # prepare output filepaths
    images_to_segment, path_segmentations, path_posteriors, path_volumes, compute = \
        prepare_output_files(path_images, path_segmentations, path_posteriors, path_volumes, recompute)

    # get label and classes lists
    label_list, n_neutral_labels = utils.get_list_labels(label_list=segmentation_label_list, FS_sort=True)
    if evaluation_label_list is None:
        evaluation_label_list = segmentation_label_list

    # prepare volume file if needed
    if path_volumes is not None:
        csv_header = [['subject'] + [str(lab) for lab in label_list[1:]]]
        with open(path_volumes, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csv_header)
        csvFile.close()

    # perform segmentation
    net = None
    previous_model_input_shape = None
    loop_info = utils.LoopInfo(len(images_to_segment), 10, 'predicting', True)
    for idx, (path_image, path_segmentation, path_posterior, tmp_compute) in enumerate(zip(images_to_segment,
                                                                                           path_segmentations,
                                                                                           path_posteriors,
                                                                                           compute)):
        # compute segmentation only if needed
        if tmp_compute:

            # preprocess image and get information
            image, aff, h, im_res, n_channels, n_dims, shape, pad_shape, crop_idx = \
                preprocess_image(path_image, n_levels, cropping, padding, aff_ref=aff_ref, dist_map=dist_map)
            model_input_shape = list(image.shape[1:])

            # prepare net for first image or if input's size has changed
            if (net is None) | (previous_model_input_shape != model_input_shape):

                # check for image size compatibility
                if (net is not None) & (previous_model_input_shape != model_input_shape) & verbose:
                    print('image of different shape as previous ones, redefining network')
                previous_model_input_shape = model_input_shape

                # build network
                net = build_model(path_model, model_input_shape, resample, im_res, n_levels, len(label_list), conv_size,
                                  nb_conv_per_level, unet_feat_count, feat_multiplier, activation, sigma_smoothing)

            if verbose:
                loop_info.update(idx)

            # predict posteriors
            prediction_patch = net.predict(image)

            # get posteriors and segmentation
            seg, posteriors = postprocess(prediction_patch, pad_shape, shape, crop_idx, n_dims, label_list,
                                          keep_biggest_component, aff, aff_ref=aff_ref,
                                          keep_biggest_of_each_group=keep_biggest_component,
                                          n_neutral_labels=n_neutral_labels)

            # write results to disk
            if path_segmentation is not None:
                utils.save_volume(seg.astype('int'), aff, h, path_segmentation)
            if path_posterior is not None:
                if n_channels > 1:
                    posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors.astype('float'), aff, h, path_posterior)

        else:
            if path_volumes is not None:
                posteriors, _, _, _, _, _, im_res = utils.get_volume_info(path_posterior, True, aff_ref=np.eye(4))
            else:
                posteriors = im_res = None

        # compute volumes
        if path_volumes is not None:
            volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
            volumes = np.around(volumes * np.prod(im_res), 3)
            row = [os.path.basename(path_image).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
            row += [np.sum(volumes[:int(len(volumes) / 2)]), np.sum(volumes[int(len(volumes) / 2):])]
            with open(path_volumes, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

    # evaluate
    if gt_folder is not None:

        # find path evaluation folder
        path_first_result = path_segmentations[0] if (path_segmentations[0] is not None) else path_posteriors[0]
        eval_folder = os.path.dirname(path_first_result)

        # compute evaluation metrics
        evaluate.dice_evaluation(gt_folder,
                                 eval_folder,
                                 evaluation_label_list,
                                 compute_distances=compute_distances,
                                 compute_score_whole_structure=False,
                                 path_dice=os.path.join(eval_folder, 'dice.npy'),
                                 path_hausdorff=os.path.join(eval_folder, 'hausdorff.npy'),
                                 path_mean_distance=os.path.join(eval_folder, 'mean_distance.npy'),
                                 recompute=recompute,
                                 verbose=verbose)


def prepare_output_files(path_images, out_seg, out_posteriors, out_volumes, recompute):

    assert out_seg or out_posteriors, "output segmentation (or posteriors) is required"

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg) if (out_seg is not None) else out_seg
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # prepare input/output volumes
    if ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):
        if os.path.isfile(path_images):
            raise Exception('extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        images_to_segment = utils.list_images_in_folder(path_images)
        if out_seg is not None:
            utils.mkdir(out_seg)
            out_seg = [os.path.join(out_seg, os.path.basename(image)).replace('.nii', '_seg.nii') for image in
                       images_to_segment]
            out_seg = [seg_path.replace('.mgz', '_seg.mgz') for seg_path in out_seg]
            out_seg = [seg_path.replace('.npz', '_seg.npz') for seg_path in out_seg]
            recompute_seg = [not os.path.isfile(path_seg) for path_seg in out_seg]
        else:
            out_seg = [out_seg] * len(images_to_segment)
            recompute_seg = [False] * len(images_to_segment)
        if out_posteriors is not None:
            utils.mkdir(out_posteriors)
            out_posteriors = [os.path.join(out_posteriors, os.path.basename(image)).replace('.nii',
                              '_posteriors.nii') for image in images_to_segment]
            out_posteriors = [posteriors_path.replace('.mgz', '_posteriors.mgz') for posteriors_path in out_posteriors]
            out_posteriors = [posteriors_path.replace('.npz', '_posteriors.npz') for posteriors_path in out_posteriors]
            recompute_post = [not os.path.isfile(path_post) for path_post in out_posteriors]
        else:
            out_posteriors = [out_posteriors] * len(images_to_segment)
            recompute_post = [out_volumes is not None] * len(images_to_segment)

    else:
        assert os.path.isfile(path_images), "files does not exist: %s " \
                                            "\nplease make sure the path and the extension are correct" % path_images
        images_to_segment = [path_images]
        if out_seg is not None:
            if ('.nii.gz' not in out_seg) & ('.nii' not in out_seg) & ('.mgz' not in out_seg) & ('.npz' not in out_seg):
                utils.mkdir(out_seg)
                filename = os.path.basename(path_images).replace('.nii', '_seg.nii')
                filename = filename.replace('mgz', '_seg.mgz')
                filename = filename.replace('.npz', '_seg.npz')
                out_seg = [os.path.join(out_seg, filename)]
            else:
                utils.mkdir(os.path.dirname(out_seg))
                out_seg = [out_seg]
            recompute_seg = [not os.path.isfile(out_seg[0])]
        else:
            out_seg = [out_seg]
            recompute_seg = [False]
        if out_posteriors is not None:
            if ('.nii.gz' not in out_posteriors) & ('.nii' not in out_posteriors) & ('.mgz' not in out_posteriors) & \
                    ('.npz' not in out_posteriors):
                utils.mkdir(out_posteriors)
                filename = os.path.basename(path_images).replace('.nii', '_posteriors.nii')
                filename = filename.replace('mgz', '_posteriors.mgz')
                filename = filename.replace('.npz', '_posteriors.npz')
                out_posteriors = [os.path.join(out_posteriors, filename)]
            else:
                utils.mkdir(os.path.dirname(out_posteriors))
                out_posteriors = [out_posteriors]
            recompute_post = [not os.path.isfile(out_posteriors[0])]
        else:
            out_posteriors = [out_posteriors]
            recompute_post = [out_volumes is not None]

    recompute_list = [recompute | re_seg | re_post for (re_seg, re_post) in zip(recompute_seg, recompute_post)]

    if out_volumes is not None:
        if out_volumes[-4:] != '.csv':
            print('out_volumes provided without csv extension. Adding csv extension to output_volumes.')
            out_volumes += '.csv'
            utils.mkdir(os.path.dirname(out_volumes))

    return images_to_segment, out_seg, out_posteriors, out_volumes, recompute_list


def preprocess_image(im_path, n_levels, crop_shape=None, padding=None, aff_ref='FS', dist_map=False):

    # read image and corresponding info
    im, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(im_path, return_volume=True)

    if padding:
        im = edit_volumes.pad_volume(im, padding_shape=padding)
        pad_shape = im.shape[:n_dims]
    else:
        pad_shape = shape

    # check that patch_shape or im_shape are divisible by 2**n_levels
    if crop_shape is not None:
        crop_shape = utils.reformat_to_list(crop_shape, length=n_dims, dtype='int')
        if not all([pad_shape[i] >= crop_shape[i] for i in range(len(pad_shape))]):
            crop_shape = [min(pad_shape[i], crop_shape[i]) for i in range(n_dims)]
        if not all([size % (2**n_levels) == 0 for size in crop_shape]):
            crop_shape = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in crop_shape]
    else:
        if not all([size % (2**n_levels) == 0 for size in pad_shape]):
            crop_shape = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in pad_shape]

    # crop image if necessary
    if crop_shape is not None:
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # align image to training axes and directions
    if n_dims > 2:
        if aff_ref == 'FS':
            aff_ref = np.array([[-1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
            im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, return_aff=False)
        elif aff_ref == 'identity':
            aff_ref = np.eye(4)
            im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=aff_ref, return_aff=False)

    # normalise image
    if n_channels == 1:
        m = np.min(im)
        M = np.max(im)
        if M == m:
            im = np.zeros(im.shape)
        else:
            im = (im - m) / (M - m)
    else:
        for i in range(im.shape[-1]):
            if (not dist_map) | (dist_map & (i % 2 == 0)):
                channel = im[..., i]
                m = np.min(channel)
                M = np.max(channel)
                if M == m:
                    im[..., i] = np.zeros(channel.shape)
                else:
                    im[..., i] = (channel - m) / (M - m)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, header, im_res, n_channels, n_dims, shape, pad_shape, crop_idx


def build_model(model_file, input_shape, resample, im_res, n_levels, n_lab, conv_size, nb_conv_per_level,
                unet_feat_count, feat_multiplier, activation, sigma_smoothing):

    assert os.path.isfile(model_file), "The provided model path does not exist."

    # initialisation
    net = None
    n_dims, n_channels = utils.get_dims(input_shape, max_channels=10)
    resample = utils.reformat_to_list(resample, length=n_dims)

    # build preprocessing model
    if resample is not None:
        im_input = KL.Input(shape=input_shape, name='pre_resample_input')
        resample_factor = [im_res[i] / float(resample[i]) for i in range(n_dims)]
        resample_shape = [utils.find_closest_number_divisible_by_m(resample_factor[i] * input_shape[i],
                          2 ** n_levels, smaller_ans=False) for i in range(n_dims)]
        resampled = nrn_layers.Resize(size=resample_shape, name='pre_resample')(im_input)
        net = Model(inputs=im_input, outputs=resampled)
        input_shape = resample_shape + [n_channels]

    # build UNet
    net = nrn_models.unet(nb_features=unet_feat_count,
                          input_shape=input_shape,
                          nb_levels=n_levels,
                          conv_size=conv_size,
                          nb_labels=n_lab,
                          name='unet',
                          prefix=None,
                          feat_mult=feat_multiplier,
                          pool_size=2,
                          use_logp=True,
                          padding='same',
                          dilation_rate_mult=1,
                          activation=activation,
                          use_residuals=False,
                          final_pred_activation='softmax',
                          nb_conv_per_level=nb_conv_per_level,
                          add_prior_layer=False,
                          add_prior_layer_reg=0,
                          layer_nb_feats=None,
                          conv_dropout=0,
                          batch_norm=-1,
                          input_model=net)
    net.load_weights(model_file, by_name=True)

    # build postprocessing model
    if (resample is not None) | (sigma_smoothing != 0):

        # get UNet output
        input_tensor = net.inputs
        last_tensor = net.output

        # resample to initial resolution
        if resample is not None:
            last_tensor = nrn_layers.Resize(size=input_shape[:-1], name='post_resample')(last_tensor)

        # smooth posteriors
        if sigma_smoothing != 0:
            last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
            last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)

        # build model
        net = Model(inputs=input_tensor, outputs=last_tensor)

    return net


def postprocess(prediction, pad_shape, im_shape, crop, n_dims, labels, keep_biggest_component,
                aff, aff_ref='FS', keep_biggest_of_each_group=True, n_neutral_labels=None):

    # get posteriors and segmentation
    post_patch = np.squeeze(prediction)

    # reset posteriors to zero outside the largest connected component of each topological class
    if keep_biggest_of_each_group:
        topology_classes = np.array([0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5])
        if n_neutral_labels != len(labels):
            left = topology_classes[n_neutral_labels:]
            topology_classes = np.concatenate([topology_classes, left + np.max(left) - np.min(left) + 1])
        unique_topology_classes = np.unique(topology_classes)
        post_patch_mask = post_patch > 0.2
        for topology_class in unique_topology_classes[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = edit_volumes.get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch[..., idx] *= tmp_mask

    # renormalise posteriors and get hard segmentation
    post_patch /= np.sum(post_patch, axis=-1)[..., np.newaxis]
    seg_patch = post_patch.argmax(-1)

    # keep biggest connected component (use it with smoothing!)
    if keep_biggest_component:
        mask = seg_patch > 0
        mask = edit_volumes.get_largest_connected_component(mask)
        seg_patch = seg_patch * mask

    # align prediction back to first orientation
    if n_dims > 2:
        if aff_ref == 'FS':
            aff_ref = np.array([[-1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
            seg_patch = edit_volumes.align_volume_to_ref(seg_patch, aff_ref, aff_ref=aff, return_aff=False)
            post_patch = edit_volumes.align_volume_to_ref(post_patch, aff_ref, aff_ref=aff, n_dims=n_dims)
        elif aff_ref == 'identity':
            aff_ref = np.eye(4)
            seg_patch = edit_volumes.align_volume_to_ref(seg_patch, aff_ref, aff_ref=aff, return_aff=False)
            post_patch = edit_volumes.align_volume_to_ref(post_patch, aff_ref, aff_ref=aff, n_dims=n_dims)

    # paste patches back to matrix of original image size
    if crop is not None:
        seg = np.zeros(shape=pad_shape, dtype='int32')
        posteriors = np.zeros(shape=[*pad_shape, labels.shape[0]])
        posteriors[..., 0] = np.ones(pad_shape)  # place background around patch
        if n_dims == 2:
            seg[crop[0]:crop[2], crop[1]:crop[3]] = seg_patch
            posteriors[crop[0]:crop[2], crop[1]:crop[3], :] = post_patch
        elif n_dims == 3:
            seg[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5]] = seg_patch
            posteriors[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], :] = post_patch
    else:
        seg = seg_patch
        posteriors = post_patch
    seg = labels[seg.astype('int')].astype('int')

    if im_shape != pad_shape:
        bounds = [int((p-i)/2) for (p, i) in zip(pad_shape, im_shape)]
        bounds += [p + i for (p, i) in zip(bounds, im_shape)]
        seg = edit_volumes.crop_volume_with_idx(seg, bounds)

    return seg, posteriors
