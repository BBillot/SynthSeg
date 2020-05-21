# python imports
import os
import csv
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model
from scipy.ndimage import label

# project imports
from SynthSeg import evaluate

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models
from ext.lab2im import edit_tensors as l2i_et


def predict(path_images,
            path_model,
            segmentation_label_list,
            path_segmentations=None,
            path_posteriors=None,
            path_volumes=None,
            voxel_volume=1.,
            skip_background_volume=True,
            padding=None,
            cropping=None,
            resample=None,
            sigma_smoothing=0,
            keep_biggest_component=False,
            conv_size=3,
            n_levels=5,
            nb_conv_per_level=2,
            unet_feat_count=24,
            feat_multiplier=2,
            no_batch_norm=False,
            gt_folder=None):
    """
    This function uses trained models to segment images.
    It is crucial that the inputs match the architecture parameters of the trained model.
    :param path_images: path of the images to segment. Can be the path to a directory or the path to a single image.
    :param path_model: path ot the trained model.
    :param segmentation_label_list: List of labels for which to compute Dice scores. It should contain the same values
    as the segmentation label list used for training the network.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array.
    :param path_segmentations: (optional) path where segmentations will be writen.
    Should be a dir, if path_images is a dir, and afile if path_images is a file.
    Should not be None, if path_posteriors is None.
    :param path_posteriors: (optional) path where posteriors will be writen.
    Should be a dir, if path_images is a dir, and afile if path_images is a file.
    Should not be None, if path_segmentations is None.
    :param path_volumes: (optional) path of a csv file where the soft volumes of all segmented regions will be writen.
    The rows of the csv file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    :param voxel_volume: (optional) volume of voxel. Default is 1 (i.e. returned volumes are voxel counts).
    :param skip_background_volume: (optional) whether to skip computing the volume of the background. This assumes the
    background correspond to the first value in label list.
    :param padding: (optional) crop the images to the specified shape before predicting the segmentation maps.
    If padding and cropping are specified, images are padded before being cropped.
    Can be an int, a sequence or a 1d numpy array.
    :param cropping: (optional) crop the images to the specified shape before predicting the segmentation maps.
    If padding and cropping are specified, images are padded before being cropped.
    Can be an int, a sequence or a 1d numpy array.
    :param resample: (optional) resample the images to the specified resolution before predicting the segmentation maps.
    Can be an int, a sequence or a 1d numpy array.
    :param sigma_smoothing: (optional) If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    :param keep_biggest_component: (optional) whether to only keep the biggest component in the predicted segmentation.
    :param conv_size: (optional) size of unet's convolution masks. Default is 3.
    :param n_levels: (optional) number of levels for unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolution layers per level. Default is 2.
    :param unet_feat_count: (optional) number of features for the first layer of the unet. Default is 24.
    :param feat_multiplier: (optional) multiplicative factor for the number of feature for each new level. Default is 2.
    :param no_batch_norm: (optional) whether to deactivate batch norm. Default is False.
    :param gt_folder: (optional) folder containing ground truth files for evaluation.
    A numpy array containing all dice scores (labels in rows, subjects in columns) will be writen either at
    segmentations_dir (if not None), or posteriors_dir.
    """

    assert path_model, "A model file is necessary"
    assert path_segmentations or path_posteriors, "output segmentation (or posteriors) is required"

    # prepare output filepaths
    images_to_segment, path_segmentations, path_posteriors, path_volumes = prepare_output_files(path_images,
                                                                                                path_segmentations,
                                                                                                path_posteriors,
                                                                                                path_volumes)

    # get label and classes lists
    label_list, _ = utils.get_list_labels(label_list=segmentation_label_list, FS_sort=True)

    # prepare volume file if needed
    if path_volumes is not None:
        if skip_background_volume:
            csv_header = [['subject'] + [str(lab) for lab in label_list[1:]]]
        else:
            csv_header = [['subject'] + [str(lab) for lab in label_list]]
        with open(path_volumes, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csv_header)
        csvFile.close()

    # perform segmentation
    net = None
    previous_model_input_shape = None
    for idx, (im_path, seg_path, posteriors_path) in enumerate(zip(images_to_segment,
                                                                   path_segmentations,
                                                                   path_posteriors)):
        utils.print_loop_info(idx, len(images_to_segment), 10)

        # preprocess image and get information
        image, aff, h, n_channels, n_dims, shape, pad_shape, cropping, crop_idx = preprocess_image(im_path,
                                                                                                   n_levels,
                                                                                                   cropping,
                                                                                                   padding)
        model_input_shape = image.shape[1:]

        # prepare net for first image or if input's size has changed
        if (idx == 0) | (previous_model_input_shape != model_input_shape):

            # check for image size compatibility
            if (idx != 0) & (previous_model_input_shape != model_input_shape):
                print('image of different shape as previous ones, redefining network')
            previous_model_input_shape = model_input_shape
            net = None

            if resample is not None:
                net, resample_shape = preprocessing_model(resample, model_input_shape, h, n_channels, n_dims, n_levels)
            else:
                resample_shape = previous_model_input_shape
            net = prepare_unet(resample_shape, len(label_list), conv_size, n_levels, nb_conv_per_level, unet_feat_count,
                               feat_multiplier, no_batch_norm, path_model, input_model=net)
            if (resample is not None) | (sigma_smoothing != 0):
                net = postprocessing_model(net, model_input_shape, resample, sigma_smoothing, n_dims)

        # predict posteriors
        prediction_patch = net.predict(image)

        # get posteriors and segmentation
        seg, posteriors = postprocess(prediction_patch, cropping, pad_shape, shape, crop_idx, n_dims, label_list,
                                      keep_biggest_component)

        # compute volumes
        if path_volumes is not None:
            if skip_background_volume:
                volumes = np.around(np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1))), 3)
            else:
                volumes = np.around(np.sum(posteriors, axis=tuple(range(0, len(posteriors.shape) - 1))), 3)
            volumes = voxel_volume * volumes
            row = [os.path.basename(im_path)] + [str(vol) for vol in volumes]
            with open(path_volumes, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

        # write results to disk
        if seg_path is not None:
            utils.save_volume(seg.astype('int'), aff, h, seg_path)
        if posteriors_path is not None:
            if n_channels > 1:
                new_shape = list(posteriors.shape)
                new_shape.insert(-1, 1)
                new_shape = tuple(new_shape)
                posteriors = np.reshape(posteriors, new_shape)
            utils.save_volume(posteriors.astype('float'), aff, h, posteriors_path)

    # evaluate
    if gt_folder is not None:
        if path_segmentations[0] is not None:
            eval_folder = os.path.dirname(path_segmentations[0])
        else:
            eval_folder = os.path.dirname(path_posteriors[0])
        path_result_dice = os.path.join(eval_folder, 'dice.npy')
        evaluate.dice_evaluation(gt_folder, eval_folder, segmentation_label_list, path_result_dice)


def prepare_output_files(path_images, out_seg, out_posteriors, out_volumes):

    # prepare input/output volumes
    if ('nii.gz' not in path_images) & ('.mgz' not in path_images) & ('.npz' not in path_images):
        images_to_segment = utils.list_images_in_folder(path_images)
        assert len(images_to_segment) > 0, "Could not find any training data"
        if out_seg:
            if not os.path.exists(out_seg):
                os.mkdir(out_seg)
            out_seg = [os.path.join(out_seg, os.path.basename(image)).replace('.nii.gz', '_seg.nii.gz') for image in
                       images_to_segment]
            out_seg = [seg_path.replace('.mgz', '_seg.mgz') for seg_path in out_seg]
            out_seg = [seg_path.replace('.npz', '_seg.npz') for seg_path in out_seg]
        else:
            out_seg = [out_seg] * len(images_to_segment)
        if out_posteriors:
            if not os.path.exists(out_posteriors):
                os.mkdir(out_posteriors)
            out_posteriors = [os.path.join(out_posteriors, os.path.basename(image)).replace('.nii.gz',
                              '_posteriors.nii.gz') for image in images_to_segment]
            out_posteriors = [posteriors_path.replace('.mgz', '_posteriors.mgz')
                              for posteriors_path in out_posteriors]
            out_posteriors = [posteriors_path.replace('.npz', '_posteriors.npz')
                              for posteriors_path in out_posteriors]
        else:
            out_posteriors = [out_posteriors] * len(images_to_segment)
        if out_volumes:
            if out_volumes[-4:] != '.csv':
                out_volumes += '.csv'
            if not os.path.exists(os.path.dirname(out_volumes)):
                os.mkdir(os.path.dirname(out_volumes))
    else:
        assert os.path.exists(path_images), "Could not find image to segment"
        images_to_segment = [path_images]
        if out_seg is not None:
            if ('nii.gz' not in out_seg) & ('.mgz' not in out_seg) & ('.npz' not in out_seg):
                if not os.path.exists(out_seg):
                    os.mkdir(out_seg)
                filename = os.path.basename(path_images).replace('.nii.gz', '_seg.nii.gz')
                filename = filename.replace('mgz', '_seg.mgz')
                filename = filename.replace('.npz', '_seg.npz')
                out_seg = os.path.join(out_seg, filename)
        out_seg = [out_seg]
        if out_posteriors is not None:
            if ('nii.gz' not in out_posteriors) & ('.mgz' not in out_posteriors) & ('.npz' not in out_posteriors):
                if not os.path.exists(out_posteriors):
                    os.mkdir(out_posteriors)
                filename = os.path.basename(path_images).replace('.nii.gz', '_posteriors.nii.gz')
                filename = filename.replace('mgz', '_posteriors.mgz')
                filename = filename.replace('.npz', '_posteriors.npz')
                out_posteriors = os.path.join(out_posteriors, filename)
        out_posteriors = [out_posteriors]

    return images_to_segment, out_seg, out_posteriors, out_volumes


def preprocess_image(im_path, n_levels, crop_shape=None, padding=None):

    # read image and corresponding info
    im, shape, aff, n_dims, n_channels, header, labels_res = utils.get_volume_info(im_path, return_volume=True)

    if padding:
        if n_channels == 1:
            im = np.pad(im, padding, mode='constant')
            pad_shape = im.shape
        else:
            im = np.pad(im, tuple([(padding, padding)] * n_dims + [(0, 0)]), mode='constant')
            pad_shape = im.shape[:-1]
    else:
        pad_shape = shape

    # check that patch_shape or im_shape are divisible by 2**n_levels
    if crop_shape is not None:
        crop_shape = utils.reformat_to_list(crop_shape, length=n_dims, dtype='int')
        if not all([pad_shape[i] >= crop_shape[i] for i in range(len(pad_shape))]):
            crop_shape = [min(pad_shape[i], crop_shape[i]) for i in range(n_dims)]
            print('cropping dimensions are higher than image size, changing cropping size to {}'.format(crop_shape))
        if not all([size % (2**n_levels) == 0 for size in crop_shape]):
            crop_shape = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in crop_shape]
    else:
        if not all([size % (2**n_levels) == 0 for size in pad_shape]):
            crop_shape = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in pad_shape]

    # crop image if necessary
    if crop_shape is not None:
        crop_idx = np.round((pad_shape - np.array(crop_shape)) / 2).astype('int')
        crop_idx = np.concatenate((crop_idx, crop_idx + crop_shape), axis=0)
        im = edit_volumes.crop_volume_with_idx(im, crop_idx=crop_idx)
    else:
        crop_idx = None

    # align image
    # ref_axes = np.array([0, 2, 1])
    # ref_signs = np.array([-1, 1, -1])
    # im_axes, img_signs = utils.get_ras_axis_and_signs(aff, n_dims=n_dims)
    # im = edit_volume.align_volume_to_ref(im, ref_axes, ref_signs, im_axes, img_signs)

    # normalise image
    m = np.min(im)
    M = np.max(im)
    if M == m:
        im = np.zeros(im.shape)
    else:
        im = (im-m)/(M-m)

    # add batch and channel axes
    if n_channels > 1:
        im = utils.add_axis(im)
    else:
        im = utils.add_axis(im, -2)

    return im, aff, header, n_channels, n_dims, shape, pad_shape, crop_shape, crop_idx


def preprocessing_model(resample, model_input_shape, header, n_channels, n_dims, n_levels):

    im_resolution = header['pixdim'][1:n_dims + 1]
    if not isinstance(resample, (list, tuple)):
        resample = [resample]
    if len(resample) == 1:
        resample = resample * n_dims
    else:
        assert len(resample) == n_dims, \
            'new_resolution must be of length 1 or n_dims ({}): got {}'.format(n_dims, len(resample))
    resample_factor = [im_resolution[i] / float(resample[i]) for i in range(n_dims)]
    pre_resample_shape = [utils.find_closest_number_divisible_by_m(resample_factor[i] * model_input_shape[i],
                          2 ** n_levels, smaller_ans=False) for i in range(n_dims)]
    resample_factor_corrected = [pre_resample_shape[i] / model_input_shape[i] for i in range(n_dims)]

    # add layers to model
    im_input = KL.Input(shape=model_input_shape, name='pre_resample_input')
    resampled = nrn_layers.Resize(zoom_factor=resample_factor_corrected, name='pre_resample')(im_input)

    # build model
    model_preprocessing = Model(inputs=im_input, outputs=resampled)

    # add channel dimensions to shapes
    pre_resample_shape = list(pre_resample_shape) + [n_channels]

    return model_preprocessing, pre_resample_shape


def prepare_unet(input_shape, n_lab, conv_size, n_levels, nb_conv_per_level, unet_feat_count, feat_multiplier,
                 no_batch_norm, model_file=None, input_model=None):
    if no_batch_norm:
        batch_norm_dim = None
    else:
        batch_norm_dim = -1
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
                          activation='elu',
                          use_residuals=False,
                          final_pred_activation='softmax',
                          nb_conv_per_level=nb_conv_per_level,
                          add_prior_layer=False,
                          add_prior_layer_reg=0,
                          layer_nb_feats=None,
                          conv_dropout=0,
                          batch_norm=batch_norm_dim,
                          input_model=input_model)
    if model_file is not None:
        net.load_weights(model_file, by_name=True)
    return net


def postprocessing_model(unet, posteriors_patch_shape, resample, sigma_smoothing, n_dims):

    # get output from unet
    input_tensor = unet.inputs
    last_tensor = unet.outputs
    if isinstance(last_tensor, list):
        last_tensor = last_tensor[0]

    # resample to original resolution
    if resample is not None:
        last_tensor = nrn_layers.Resize(size=posteriors_patch_shape[:-1], name='post_resample')(last_tensor)

    # smooth posteriors
    if sigma_smoothing != 0:

        # separate image channels from labels channels
        n_labels = last_tensor.get_shape().as_list()[-1]
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_labels, axis=-1), name='resample_split')(last_tensor)

        # create gaussian blurring kernel
        sigma_smoothing = utils.reformat_to_list(sigma_smoothing, length=n_dims)
        kernels_list = l2i_et.get_gaussian_1d_kernels(sigma_smoothing)

        # blur each image channel separately
        last_tensor = l2i_et.blur_tensor(split[0], kernels_list, n_dims)
        for i in range(1, n_labels):
            temp_blurred = l2i_et.blur_tensor(split[i], kernels_list, n_dims)
            last_tensor = KL.concatenate([last_tensor, temp_blurred], axis=-1, name='cat_blurring_%s' % i)

    # build model
    model_postprocessing = Model(inputs=input_tensor, outputs=last_tensor)

    return model_postprocessing


def postprocess(prediction, crop_shape, pad_shape, im_shape, crop, n_dims, labels, keep_biggest_component):

    # get posteriors and segmentation
    post_patch = np.squeeze(prediction)
    seg_patch = post_patch.argmax(-1)

    # align prediction back to first orientation
    # ref_axes = np.array([0, 2, 1])
    # ref_signs = np.array([-1, 1, -1])
    # seg_patch = edit_volume.align_volume_to_ref(seg_patch, im_axes, im_signs, ref_axes, ref_signs, n_dims)

    # keep biggest connected component (use it with smoothing!)
    if keep_biggest_component:
        components, n_components = label(seg_patch)
        if n_components > 1:
            unique_components = np.unique(components)
            size = 0
            mask = None
            for comp in unique_components[1:]:
                tmp_mask = components == comp
                tmp_size = np.sum(tmp_mask)
                if tmp_size > size:
                    size = tmp_size
                    mask = tmp_mask
            seg_patch[np.logical_not(mask)] = 0

    # paste patches back to matrix of original image size
    if crop_shape is not None:
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
        lower_bound = [int((p-i)/2) for (p, i) in zip(pad_shape, im_shape)]
        upper_bound = [p-int((p-i)/2) for (p, i) in zip(pad_shape, im_shape)]
        if n_dims == 2:
            seg = seg[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]
        elif n_dims == 3:
            seg = seg[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]]

    return seg, posteriors
