"""This file regroups all the major pre/postprocessing steps used in the SynthSeg-lesion paper.
The following list regroups all the functions of this file:
    -dilate_lesions
    -paste_lesions_on_buckner
    -downsample_upsample_flair
    -inter_rater_reproducibility_cross_val_exp
    -build_longitudinal_consensus
    -preprocess_asegs
    -validation_on_dilated_lesions
    -run_validation_on_aseg_gt
    -postprocess_samseg
    -cross_validate_posteriors_threshold
    -plot_experiment_1
    -plot_experiment_2
"""

# python imports
import os
import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_dilation, binary_erosion

# project imports
from SynthSeg.boxplots import draw_boxplots
from SynthSeg.evaluate import dice_evaluation, fast_dice

# third-party imports
from ext.lab2im import edit_volumes, utils


def dilate_lesions(labels_dir, result_dir, recompute=True):

    utils.mkdir(result_dir)

    path_labels = utils.list_images_in_folder(labels_dir)
    for path_label in path_labels:
        path_result_label = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result_label)) | recompute:

            label, aff, h = utils.load_volume(path_label, im_only=False)

            # define lesion, WM, and LV masks
            WM = (label == 2) | (label == 41)
            lesion = label == 77
            LV_and_lesion = (label == 4) | lesion

            # morphological operations to bridge the gaps between lesions and LV
            morph_struct = utils.build_binary_structure(2, len(WM.shape))
            LV_and_lesion = binary_dilation(LV_and_lesion, morph_struct)
            LV_and_lesion = binary_erosion(LV_and_lesion, morph_struct)
            lesion = (LV_and_lesion & WM) | lesion
            label[lesion] = 77

            # save new label maps
            utils.save_volume(label, aff, h, path_result_label)


def paste_lesions_on_buckner(lesion_dir, buckner_dir, result_dir, dilate=2, recompute=False):

    path_lesions = utils.list_images_in_folder(lesion_dir)
    path_buckners = utils.list_images_in_folder(buckner_dir)

    utils.mkdir(result_dir)

    # loop over buckner label maps
    for idx_buckner, path_buckner in enumerate(path_buckners):
        utils.print_loop_info(idx_buckner, len(path_buckners), 1)
        buckner_name = os.path.basename(path_buckner).replace('_seg', '').replace('.nii.gz', '')
        buckner = utils.load_volume(path_buckner)
        WM = (buckner == 2) | (buckner == 7) | (buckner == 16) | (buckner == 41) | (buckner == 46)

        # loop over challenge data
        for path_lesion in path_lesions:
            lesion_name = os.path.basename(path_lesion).replace('.samseg_and_lesions.nii.gz', '')
            path_result = os.path.join(result_dir, buckner_name + '_' + lesion_name + '.nii.gz')
            if (not os.path.isfile(path_result)) | recompute:

                lesion = utils.load_volume(path_lesion)
                assert lesion.shape == buckner.shape, 'lesions should have same shape as buckner labels'

                # define lesion, WM, and LV masks
                lesion = (lesion == 77) & WM
                LV_and_lesion = (buckner == 4) | lesion

                # morphological operations to bridge the gaps between lesions and LV
                morph_struct = utils.build_binary_structure(dilate, len(lesion.shape))
                lesion = binary_dilation(LV_and_lesion, morph_struct)
                lesion = binary_erosion(lesion, morph_struct)
                lesion = lesion & WM
                buckner_lesions = np.where(lesion, 77, buckner)

                # save map
                utils.save_volume(buckner_lesions, None, None, path_result)


def downsample_upsample_flair(flair_image_dir, recompute=True):

    # downsample images
    downsample_image_dir = flair_image_dir + '_downsampled_1_1_2.2'
    utils.mkdir(downsample_image_dir)
    edit_volumes.mri_convert_images_in_dir(flair_image_dir,
                                           downsample_image_dir,
                                           interpolation='nearest',
                                           voxsize=[1, 1, 2.2],
                                           recompute=recompute)

    # upsample images
    upsample_image_dir = flair_image_dir + '_resampled_back_to_1_1_1'
    utils.mkdir(upsample_image_dir)
    edit_volumes.mri_convert_images_in_dir(downsample_image_dir,
                                           upsample_image_dir,
                                           reference_dir=flair_image_dir,
                                           recompute=recompute)


def inter_rater_reproducibility_cross_val_exp(manual_seg_dir, ref_image_dir=None, recompute=True):

    # list subjects
    list_subjects = utils.list_subfolders(manual_seg_dir)

    # create result directories
    if ref_image_dir is not None:
        realigned_seg_dir = os.path.join(os.path.dirname(manual_seg_dir), 'registered_to_t1')
        list_ref_subjects = utils.list_images_in_folder(ref_image_dir)
    else:
        realigned_seg_dir = os.path.join(os.path.dirname(manual_seg_dir), 'realigned')
        list_ref_subjects = [None] * len(list_subjects)
    utils.mkdir(realigned_seg_dir)
    path_dice = os.path.join(realigned_seg_dir, 'dice.npy')

    # loop over subjects
    dice = list()
    if (not os.path.isfile(path_dice)) | recompute:
        for subject_dir, ref_subject in zip(list_subjects, list_ref_subjects):

            # align all images to first image
            if ref_subject is not None:
                ref_image = ref_subject
            else:
                ref_image = utils.list_images_in_folder(subject_dir)[0]
            result_dir = os.path.join(realigned_seg_dir, os.path.basename(subject_dir))
            edit_volumes.mri_convert_images_in_dir(subject_dir,
                                                   result_dir,
                                                   interpolation='nearest',
                                                   reference_dir=ref_image,
                                                   same_reference=True,
                                                   recompute=recompute)

            # load all volumes and compute distance maps
            list_segs = [utils.load_volume(path) for path in utils.list_images_in_folder(result_dir)]
            list_distance_maps = [edit_volumes.compute_distance_map(labels, crop_margin=20) for labels in list_segs]
            distance_maps = np.stack(list_distance_maps, axis=-1)
            n_raters = len(list_segs)

            # compare each segmentation to the consensus of all others
            tmp_dice = list()
            for i, seg in enumerate(list_segs):
                tmp_distance_maps = distance_maps[..., np.arange(n_raters) != i]
                tmp_distance_maps = (np.mean(tmp_distance_maps, axis=-1) > 0) * 1
                seg = (seg > 0) * 1
                tmp_dice.append(2 * np.sum(tmp_distance_maps * seg) / (np.sum(tmp_distance_maps) + np.sum(seg)))
            dice.append(tmp_dice)

        np.save(path_dice, np.array(dice))


def build_longitudinal_consensus(labels_dir_1, labels_dir_2, result_dir, recompute=True):

    # create result dir
    utils.mkdir(result_dir)

    # list all segmentations
    path_labels_1 = utils.list_files(labels_dir_1)
    path_labels_2 = utils.list_files(labels_dir_2)

    for path_lab_1, path_lab_2 in zip(path_labels_1, path_labels_2):

        # check if result is already saved
        path_result = os.path.join(result_dir, os.path.basename(path_lab_1))
        if (not os.path.isfile(path_result)) | recompute:

            # load volumes
            lab_1, aff, h = utils.load_volume(path_lab_1, im_only=False)
            lab_2 = utils.load_volume(path_lab_2)

            # compute and save consensus
            dist_masp_1 = edit_volumes.compute_distance_map(lab_1, crop_margin=20)
            dist_masp_2 = edit_volumes.compute_distance_map(lab_2, crop_margin=20)
            consensus = (np.mean(np.stack([dist_masp_1, dist_masp_2], axis=-1), axis=-1) > 0) * 1
            utils.save_volume(consensus, aff, h, path_result)


def preprocess_asegs(aseg_dir, lesion_gt_dir, list_incorrect, list_correct, lesion_label_in_gt=77, dilate=2,
                     recompute=False):

    # align asegs to gt dir (cropping to same dimension)
    cropped_dir = aseg_dir + '_cropped'
    edit_volumes.mri_convert_images_in_dir(aseg_dir,
                                           cropped_dir,
                                           interpolation='nearest',
                                           reference_dir=lesion_gt_dir,
                                           recompute=recompute)

    # correct for aseg labels
    corrected_dir = cropped_dir + '_corrected'
    edit_volumes.correct_labels_in_dir(cropped_dir, list_incorrect, list_correct, corrected_dir, smooth=False,
                                       recompute=recompute)

    # list gt and aseg, and create result dir
    list_lesion_labels = utils.list_images_in_folder(lesion_gt_dir)
    list_aseg_labels = utils.list_images_in_folder(corrected_dir)
    inpainted_dir = corrected_dir + '_lesion_inpainted'
    utils.mkdir(inpainted_dir)

    # loop over subjects
    for path_lesion_label, path_aseg_label in zip(list_lesion_labels, list_aseg_labels):
        path_result = os.path.join(inpainted_dir, os.path.basename(path_aseg_label))
        if (not os.path.isfile(path_result)) | recompute:

            # paste lesion label
            lesions = utils.load_volume(path_lesion_label)
            aseg_label, aff, h = utils.load_volume(path_aseg_label, im_only=False)
            lesion_mask = lesions == lesion_label_in_gt
            aseg_label[lesion_mask] = 77
            utils.save_volume(aseg_label, aff, h, path_result)

    # dilate lesion and ventricle
    dilated_dir = inpainted_dir + '_dilated'
    utils.mkdir(dilated_dir)
    list_inpainted_aseg = utils.list_images_in_folder(inpainted_dir)
    for path_aseg in list_inpainted_aseg:

        path_result = os.path.join(dilated_dir, os.path.basename(path_aseg))
        if (not os.path.isfile(path_result)) | recompute:

            # define lesion, WM, and LV masks
            aseg, aff, h = utils.load_volume(path_aseg, im_only=False)
            WM = aseg == 2
            LV = aseg == 4
            lesion = aseg == 77

            # morphological operations to bridge the gaps between lesions and LV
            morph_struct = utils.build_binary_structure(dilate, len(aseg.shape))
            dilated_LV_or_lesion = binary_dilation(LV | lesion, morph_struct)
            filled_LV_or_lesion = binary_erosion(dilated_LV_or_lesion, morph_struct)
            LV = LV | (filled_LV_or_lesion & WM)
            aseg[LV] = 4

            # save map
            utils.save_volume(aseg, aff, h, path_result)


def validation_on_dilated_lesions(normal_validation_dir, dilated_validation_dir, gt_dir, evaluation_labels,
                                  recompute=True):

    utils.mkdir(dilated_validation_dir)

    list_validation_subdir = utils.list_subfolders(normal_validation_dir)
    for val_idx, validation_subdir in enumerate(list_validation_subdir):
        utils.print_loop_info(val_idx, len(list_validation_subdir), 5)

        # dilate lesion
        dilated_validation_subdir = os.path.join(dilated_validation_dir, os.path.basename(validation_subdir))
        dilate_lesions(validation_subdir, dilated_validation_subdir, recompute=recompute)

        # compute new dice scores
        path_dice = os.path.join(dilated_validation_subdir, 'dice.npy')
        if (not os.path.isfile(path_dice)) | recompute:
            dice_evaluation(gt_dir, dilated_validation_subdir, evaluation_labels, path_dice)


def run_validation_on_aseg_gt(list_supervised_model_dir, list_aseg_gt_dir, path_label_list, recompute=False):
    list_main_samseg_validation_dir = [os.path.join(p, 'validation_samseg') for p in list_supervised_model_dir]

    # loop over architectures
    for (main_samseg_validation_dir, gt_dir) in zip(list_main_samseg_validation_dir, list_aseg_gt_dir):

        # list model subdirs
        main_aseg_validation_dir = os.path.join(os.path.dirname(main_samseg_validation_dir), 'validation')
        utils.mkdir(main_aseg_validation_dir)
        list_samseg_validation_subdir = utils.list_subfolders(main_samseg_validation_dir)

        # lover over models
        for samseg_validation_subdir in list_samseg_validation_subdir:

            # create equivalent aseg subdir
            aseg_validation_subdir = os.path.join(main_aseg_validation_dir, os.path.basename(samseg_validation_subdir))
            utils.mkdir(aseg_validation_subdir)
            path_aseg_dice = os.path.join(aseg_validation_subdir, 'dice.npy')

            # compute dice with aseg gt
            if (not os.path.isfile(path_aseg_dice)) | recompute:
                dice_evaluation(gt_dir,
                                samseg_validation_subdir,
                                path_label_list,
                                path_aseg_dice)


def postprocess_samseg(list_samseg_dir,
                       list_gt_dir,
                       path_segmentation_labels,
                       incorrect_labels,
                       correct_labels,
                       list_posteriors_dir=None,
                       list_thresholds=None,
                       recompute=False):

    """ This function processes the samseg segmentations: it corrects the labels (right/left and 99 to 77), resamples
    them to the space of gt_dir, and computes the Dice scores for 1) all_subjects vs. testing subjects only, and 2) all
    ROIs vs. lesions only.
    It requires that all segmentations are sorted in three subfolders inside samseg_main_dir: t1, flair, and t1_flair.
    IMPORTANT: Images are expected to have to following naming convention: <subject_id>.samseg.<contrast>.lesion.mgz,
    where <contrast> must either be t1, flair, ***t1_flair***
    :param list_samseg_dir: main samseg dir containing the three subfolders t1, flair, t1_flair
    :param list_gt_dir: folder with the gt label maps for all subjects
    :param path_segmentation_labels: list of segmentation labels
    :param incorrect_labels: list of samseg incorrect labels
    :param correct_labels: list of labels to correct the wrong one with
    :param recompute: whether to recompute files
    """

    if list_posteriors_dir is None:
        list_posteriors_dir = [None] * len(list_samseg_dir)

    for samseg_dir, gt_dir, posteriors_dir, threshold in zip(list_samseg_dir, list_gt_dir, list_posteriors_dir,
                                                             list_thresholds):

        # define result directories
        samseg_corrected_dir = samseg_dir + '_corrected'
        samseg_preprocessed_dir = samseg_dir + '_preprocessed'
        if (not os.path.isdir(samseg_preprocessed_dir)) | recompute:

            # regroup right/left labels and change 99 to 77
            edit_volumes.correct_labels_in_dir(samseg_dir,
                                               incorrect_labels,
                                               correct_labels,
                                               samseg_corrected_dir,
                                               recompute=recompute)

            # resample to gt format
            edit_volumes.mri_convert_images_in_dir(samseg_corrected_dir,
                                                   samseg_preprocessed_dir,
                                                   interpolation='nearest',
                                                   reference_dir=gt_dir,
                                                   recompute=recompute)

        # replace lesions by thresholded lesion posteriors
        if posteriors_dir is not None:

            # resample posteriors to gt format
            posteriors_preprocessed_dir = posteriors_dir + '_preprocessed'
            edit_volumes.mri_convert_images_in_dir(posteriors_dir,
                                                   posteriors_preprocessed_dir,
                                                   reference_dir=gt_dir,
                                                   recompute=recompute)

            # list hard segmentations and posteriors
            samseg_postprocessed_dir = samseg_dir + '_postprocessed'
            utils.mkdir(samseg_postprocessed_dir)
            path_segs = [path for path in utils.list_images_in_folder(samseg_preprocessed_dir)]
            path_posteriors = [path for path in utils.list_images_in_folder(posteriors_preprocessed_dir)]

            for subject_idx, (path_seg, path_post) in enumerate(zip(path_segs, path_posteriors)):
                path_result = os.path.join(samseg_postprocessed_dir, os.path.basename(path_seg))
                if (not os.path.isfile(path_result)) | recompute:

                    # replace segmented lesions by thresholded posteriors
                    seg, aff, h = utils.load_volume(path_seg, im_only=False)
                    posteriors = utils.load_volume(path_post)
                    seg[seg == 77] = 2
                    seg[posteriors > threshold] = 77
                    utils.save_volume(seg, aff, h, path_result)

        else:
            samseg_postprocessed_dir = samseg_preprocessed_dir

        # compute dice scores with
        path_dice_testing = os.path.join(samseg_postprocessed_dir, 'dice.npy')
        path_dice_lesions_testing = os.path.join(samseg_postprocessed_dir, 'dice_lesions.npy')
        if (not os.path.isfile(path_dice_testing)) | recompute:
            dice_evaluation(gt_dir, samseg_postprocessed_dir, path_segmentation_labels, path_dice_testing)
        if (not os.path.isfile(path_dice_lesions_testing)) | recompute:
            dice = np.load(path_dice_testing)
            np.save(path_dice_lesions_testing, dice[4, :])


def cross_validate_posteriors_threshold(list_seg_dir, list_posteriors_dir, list_gt_dir, list_thresholds,
                                        recompute=True):

    for fold_idx, (seg_dir, posteriors_dir, gt_dir) in enumerate(zip(list_seg_dir, list_posteriors_dir, list_gt_dir)):

        path_dice = os.path.join(os.path.dirname(seg_dir), 'dice_lesions_for_thresholds.npy')
        path_dice_means = os.path.join(os.path.dirname(seg_dir), 'dice_lesions_means_for_thresholds.npy')
        if (not os.path.isfile(path_dice)) | (not os.path.isfile(path_dice_means)) | recompute:

            path_segs = [path for path in utils.list_images_in_folder(seg_dir)]
            path_posteriors = [path for path in utils.list_images_in_folder(posteriors_dir)]
            path_gts = [path for path in utils.list_images_in_folder(gt_dir)]
            dice = np.zeros((len(list_thresholds), len(path_gts)))

            for subject_idx, (path_seg, path_post, path_gt) in enumerate(zip(path_segs, path_posteriors, path_gts)):

                seg = utils.load_volume(path_seg)
                posteriors = utils.load_volume(path_post)
                gt = utils.load_volume(path_gt)

                seg[seg == 77] = 2
                for idx, threshold in enumerate(list_thresholds):
                    tmp_seg = deepcopy(seg)
                    lesion_mask = posteriors > threshold
                    tmp_seg[lesion_mask] = 77
                    dice[idx, subject_idx] = fast_dice(gt, tmp_seg, [77])

            np.save(path_dice, dice)
            np.save(path_dice_means, np.mean(dice, axis=1))

        dice_means = np.load(path_dice_means)
        max_threshold = list_thresholds[np.argmax(dice_means)]
        print('max threshold for fold {0}: {1:.2f}'.format(fold_idx, max_threshold))


def plot_experiment_1(list_methods_segm, indices):

    list_contrast = ['T1', 'FLAIR', 'T1-FLAIR']

    my_palette = {'manual': (0.7878969627066512, 0.09217993079584776, 0.11200307574009996),
                  'supervised': (0.9137254901960784, 0.3686274509803921, 0.050980392156862744),
                  'SAMSEG-lesion': (0.9921568627450981, 0.5793310265282584, 0.272879661668589),
                  'SAMSEG-lesion-retrained': (0.6083967704728951, 0.19538638985005768, 0.012856593617839291),
                  'SynthSeg': (0.08404459823144944, 0.38506728181468663, 0.6644367550941945),
                  'SynthSeg-mix': (0.2758477508650519, 0.5841753171856978, 0.783114186851211),
                  'SynthSeg-rule': (0.647289504036909, 0.803921568627451, 0.8920415224913495)}

    # list all Dice scores to plot
    list_dice_lesion = list()
    list_datasets_lesion = list()
    list_methods_lesion = list()
    list_dice = list()
    list_datasets = list()
    list_methods = list()
    for contrast in list_contrast:

        for idx, methods_segm in enumerate(list_methods_segm):

            # if idx > 0:
            list_dice.append(os.path.join(methods_segm, contrast, 'dice_means_folds_1_2_3.npy'))
            list_datasets.append(contrast)
            list_methods.append(os.path.basename(methods_segm))
            list_dice_lesion.append(os.path.join(methods_segm, contrast, 'dice_lesions_folds_1_2_3.npy'))
            list_datasets_lesion.append(contrast)
            list_methods_lesion.append(os.path.basename(methods_segm))

    draw_boxplots(list_dice_lesion,
                  list_datasets_lesion,
                  list_methods_lesion,
                  None,
                  indices,
                  palette=my_palette,
                  order=list_contrast,
                  figsize=(6, 4.5),
                  legend_loc='lower left',
                  remove_legend=True,
                  fontsize=24,
                  fontsize_legend=13,
                  y_label=None,
                  boxplot_width=0.8,
                  boxplot_linewidth=2,
                  outlier_size=6,
                  av_plot_ylim=[0, 0.85],
                  av_plot_step_yticks=0.2,
                  av_plot_title='Dice for MS lesions',
                  draw_subplots=False,
                  path_av_figure='/home/benjamin/data/lesions/MSSeg/segmentations/exp1_lesion_dice.pdf')
    draw_boxplots(list_dice,
                  list_datasets,
                  list_methods,
                  None,
                  indices,
                  palette=my_palette,
                  order=list_contrast,
                  figsize=(6, 4.5),
                  legend_loc='lower left',
                  remove_legend=True,
                  fontsize=24,
                  fontsize_legend=13,
                  y_label=None,
                  boxplot_width=0.85,
                  boxplot_linewidth=2,
                  outlier_size=6,
                  av_plot_ylim=[0.74, 0.915],
                  av_yticks=np.array([0.75, 0.8, 0.85, 0.90]),
                  av_plot_title='Average Dice for brain ROIs',
                  draw_subplots=False,
                  path_av_figure='/home/benjamin/data/lesions/MSSeg/segmentations/exp1_dice.pdf')


def plot_experiment_2(list_methods_segm, indices):

    list_contrast = ['T1', 'FLAIR', 'T1-FLAIR']

    my_palette = {'manual': (0.7878969627066512, 0.09217993079584776, 0.11200307574009996),
                  'supervised': (0.9137254901960784, 0.3686274509803921, 0.050980392156862744),
                  'SAMSEG-lesion': (0.9921568627450981, 0.5793310265282584, 0.272879661668589),
                  'SAMSEG-FS':  (0.6083967704728951, 0.19538638985005768, 0.012856593617839291),
                  'SynthSeg': (0.08404459823144944, 0.38506728181468663, 0.6644367550941945),
                  'SynthSeg-mix': (0.2758477508650519, 0.5841753171856978, 0.783114186851211),
                  'SynthSeg-rule': (0.647289504036909, 0.803921568627451, 0.8920415224913495)}

    # list all Dice scores to plot
    list_dice_lesion = list()
    list_datasets_lesion = list()
    list_methods_lesion = list()
    list_dice = list()
    list_datasets = list()
    list_methods = list()
    for contrast in list_contrast:

        for idx, methods_segm in enumerate(list_methods_segm):

            # if idx > 0:
            list_dice.append(os.path.join(methods_segm, contrast, 'dice_means.npy'))
            list_datasets.append(contrast)
            list_methods.append(os.path.basename(methods_segm))
            list_dice_lesion.append(os.path.join(methods_segm, contrast, 'dice_lesions.npy'))
            list_datasets_lesion.append(contrast)
            list_methods_lesion.append(os.path.basename(methods_segm))

    draw_boxplots(list_dice_lesion,
                  list_datasets_lesion,
                  list_methods_lesion,
                  None,
                  indices,
                  palette=my_palette,
                  order=list_contrast,
                  figsize=(6, 4.5),
                  legend_loc='lower right',
                  remove_legend=True,
                  fontsize=24,
                  fontsize_legend=13,
                  y_label=None,
                  boxplot_width=0.75,
                  boxplot_linewidth=2,
                  outlier_size=6,
                  av_plot_ylim=[0, 0.84],
                  av_plot_step_yticks=0.2,
                  av_plot_title='Dice for MS lesions',
                  draw_subplots=False,
                  path_av_figure='/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/exp2_lesion_dice.pdf')
    draw_boxplots(list_dice,
                  list_datasets,
                  list_methods,
                  None,
                  indices,
                  palette=my_palette,
                  order=list_contrast,
                  figsize=(6, 4.5),
                  legend_loc=(0.422, 0.01),
                  remove_legend=False,
                  fontsize=24,
                  fontsize_legend=18.7,
                  y_label=None,
                  boxplot_width=0.75,
                  boxplot_linewidth=2,
                  outlier_size=6,
                  av_plot_ylim=[0.55, 0.835],
                  av_yticks=np.array([0.6, 0.7, 0.8]),
                  av_plot_title='Average Dice for brain ROIs',
                  draw_subplots=False,
                  path_av_figure='/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/exp2_dice.pdf')


if __name__ == '__main__':

    # --------------------- create anisotropic real data for ISBI longitudinal supervised networks ---------------------

    downsample_upsample_flair(flair_image_dir='/home/benjamin/data/lesions/MSSeg/images/resampled_1_1_1/flair_cropped')
    edit_volumes.create_mutlimodal_images(
        ['/home/benjamin/data/lesions/MSSeg/images/resampled_1_1_1/t1_cropped',
         '/home/benjamin/data/lesions/MSSeg/images/resampled_1_1_1/flair_cropped_resampled_back_to_1_1_1'],
        '/home/benjamin/data/lesions/MSSeg/images/resampled_1_1_1/t1_resampled_flair_cropped')

    # -------------------------------- inter-rater reprodicibility MSSeg cross-validation ------------------------------

    # compute inter-rater repro in cross_validation experiement
    original_seg_dir = '/home/benjamin/data/lesions/MSSeg/labels/initial_resolution/lesion_masks'
    ref_dir = '/home/benjamin/data/lesions/MSSeg/labels/resample_1_1_1/aseg_lesions_gt/aseg_lesions'
    inter_rater_reproducibility_cross_val_exp(original_seg_dir, ref_dir, recompute=False)

    # --------------------------------------- compute consensus ISBI longitudinal --------------------------------------

    # compute consensus segmentations for longitudinal data
    labels_folder_1 = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_masks/lesion_mask1'
    labels_folder_2 = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_masks/lesion_mask2'
    result_folder = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_consensus/lesion_consensus'
    build_longitudinal_consensus(labels_folder_1, labels_folder_2, result_folder, recompute=True)

    # ----------------------------------- inter-rater reprodicibility ISBI longitudinal --------------------------------

    gt_folder = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_masks/lesion_mask1_2.2_only'
    seg_dir = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_masks/lesion_mask2_2.2_only'
    path_result_dice = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_masks/dice_1_vs_2_2.2_only.npy'
    dice_evaluation(gt_dir=gt_folder, seg_dir=seg_dir, path_label_list=[1], path_result_dice_array=path_result_dice,
                    cropping_margin_around_gt=None)

    # ---------------------------------------------- preprocess MSSeg asegs --------------------------------------------

    aseg_folder = '/home/benjamin/data/lesions/MSSeg/labels/resample_1_1_1/asegs'
    cropped_labels_dir = '/home/benjamin/data/lesions/MSSeg/labels/resample_1_1_1/SAMSEG_generation/samseg_lesions'
    main_result_dir = '/home/benjamin/data/lesions/MSSeg/labels/resample_1_1_1/aseg_lesions'
    incorrect = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/aseg_incorrect_labels.npy'
    correct = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/aseg_correct_labels.npy'
    preprocess_asegs(aseg_folder, cropped_labels_dir, incorrect, correct)

    # --------------------------------------- preprocess ISBI longitudinal asegs ---------------------------------------

    aseg_folder = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/asegs/asegs_original'
    consensus_folder = '/home/benjamin/data/lesions/ISBI_longitudinal/labels/lesion_consensus/lesion_consensus'
    incorrect = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/aseg_incorrect_labels.npy'
    correct = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/aseg_correct_labels.npy'
    preprocess_asegs(aseg_folder, consensus_folder, incorrect, correct, lesion_label_in_gt=1, recompute=False)

    # ------------------------------------- process SAMSEG-lesion MSSeg ------------------------------------

    # data dir
    samseg_dirs = ['/home/benjamin/data/lesions/MSSeg/SAMSEG/SAMSEG-lesion/flair',
                   '/home/benjamin/data/lesions/MSSeg/SAMSEG/SAMSEG-lesion/t1',
                   '/home/benjamin/data/lesions/MSSeg/SAMSEG/SAMSEG-lesion/t1_flair']
    gt_folders = ['/home/benjamin/data/lesions/MSSeg/labels/resample_1_1_1/aseg_lesions_gt/aseg_lesions'] * 3
    # labels
    incorrect = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/samseg_incorrect_labels_generation.npy'
    correct = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/samseg_correct_labels_generation.npy'
    segmentation_labels = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/segmentation_labels.npy'
    # postprocess
    postprocess_samseg(samseg_dirs, gt_folders, segmentation_labels, incorrect, correct, recompute=False)

    # ------------------------------------- process SAMSEG-lesion ISBI longitudinal ------------------------------------

    # data dir
    samseg_dirs = ['/home/benjamin/data/lesions/ISBI_longitudinal/SAMSEG/SAMSEG-lesions/flair/flair_2.2_only',
                   '/home/benjamin/data/lesions/ISBI_longitudinal/SAMSEG/SAMSEG-lesions/t1/t1_2.2_only',
                   '/home/benjamin/data/lesions/ISBI_longitudinal/SAMSEG/SAMSEG-lesions/t1_flair/t1_flair_2.2_only']
    gt_folders = ['/home/benjamin/data/lesions/ISBI_longitudinal/labels/asegs/asegs_lesions_2.2_only'] * 3
    # labels
    incorrect = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/samseg_incorrect_labels_generation.npy'
    correct = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/samseg_correct_labels_generation.npy'
    segmentation_labels = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/segmentation_labels.npy'
    # postprocess
    postprocess_samseg(samseg_dirs, gt_folders, segmentation_labels, incorrect, correct, recompute=False)

    # --------------------------------- plot results first experiment with FS version ----------------------------------

    # competing methods
    list_methods_segmentations = [
        '/home/benjamin/data/lesions/MSSeg/segmentations/supervised',
        '/home/benjamin/data/lesions/MSSeg/segmentations/SAMSEG-lesion',
        '/home/benjamin/data/lesions/MSSeg/segmentations/SynthSeg',
        '/home/benjamin/data/lesions/MSSeg/segmentations/SynthSeg-rule',
        '/home/benjamin/data/lesions/MSSeg/segmentations/SynthSeg-mix']
    # plotting options
    eval_indices = '/home/benjamin/data/lesions/MSSeg/labels_classes_stats/eval_indices_lesion_less.npy'
    # plot
    plot_experiment_1(list_methods_segmentations, eval_indices)

    # --------------------------------- plot results second experiment with FS version ---------------------------------

    # competing methods
    list_methods_segmentations = [
        '/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/supervised',
        '/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/SAMSEG-lesion',
        '/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/SynthSeg',
        '/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/SynthSeg-rule',
        '/home/benjamin/data/lesions/ISBI_longitudinal/segmentations/SynthSeg-mix']
    # plotting options
    eval_indices = '/home/benjamin/data/lesions/ISBI_longitudinal/labels_classes_stats/eval_indices_lesion_less.npy'
    # plot
    plot_experiment_2(list_methods_segmentations, eval_indices)
