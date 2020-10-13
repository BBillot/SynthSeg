"""This file regroups all the major pre/postprocessing steps used in the PV-SynthSeg paper."""

# python imports
import os
import numpy as np

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def prepare_anisotropic_dataset(image_dir,
                                list_synth_res,
                                downsample_image_result_dir,
                                resample_image_result_dir,
                                labels_dir=None,
                                downsample_labels_result_dir=None,
                                native_resolution=1,
                                slice_thickness=4,
                                recompute=True):
    """
    This function takes as input a set of isotropic HR images (e.g. 1mm) and a list of resolutions.
    The HR images will be downsampled to each resolution but only in one of the three direction at a time (e.g. 1*1*r,
    then 1*r*1, then r*1*1). All downsampled images are then resampled back to native HR.
    Additionally, this function can also downsample label maps corresponding to the input images.
    :param image_dir: path of directory with input images
    :param list_synth_res: list of low resolutions (e.g. [3, 6, 9]).
    :param downsample_image_result_dir: path of directory where all versions of downsampled images will be writen.
    :param resample_image_result_dir: path of directory where all versions of resampled images will be writen.
    :param native_resolution: native HR (useful when all HR images are not exactly the same resolution)
    :param slice_thickness: slice thickness (same for all resolution)
    :param labels_dir: path of directory with label maps corresponding to input images
    :param downsample_labels_result_dir: path of directory where all versions of downsampled labels will be writen.
    :param recompute:
    """

    # create results dir
    utils.mkdir(resample_image_result_dir)
    utils.mkdir(downsample_image_result_dir)
    if downsample_labels_result_dir is not None:
        utils.mkdir(downsample_labels_result_dir)

    # define thickness, which is always the same
    list_thickness = slice_thickness * np.eye(3)
    list_thickness[list_thickness == 0] = native_resolution
    list_thickness = list_thickness.tolist()

    # loop over resolution levels
    for synth_res in list_synth_res:

        # define blurring res
        list_data_res = synth_res * np.eye(3)
        list_data_res[list_data_res == 0] = native_resolution
        list_data_res = list_data_res.tolist()

        # loop over resolution directions
        for (data_res, thickness) in zip(list_data_res, list_thickness):
            res_str = '_'.join(['%d' % r for r in data_res])
            print('\npreprocess images to ' + res_str.replace('_', '*') + 'mm resolution')

            # build path result folders
            im_results_dir = os.path.join(resample_image_result_dir, 'images_' + res_str)
            im_downsample_results_dir = os.path.join(downsample_image_result_dir, 'images_' + res_str)
            if downsample_labels_result_dir is not None:
                labels_results_dir = os.path.join(downsample_labels_result_dir, 'labels_' + res_str)
            else:
                labels_results_dir = None

            # downsample datasets
            edit_volumes.simulate_upsampled_anisotropic_images(image_dir,
                                                               im_downsample_results_dir,
                                                               im_results_dir,
                                                               data_res,
                                                               labels_dir=labels_dir,
                                                               downsample_labels_result_dir=labels_results_dir,
                                                               slice_thickness=thickness,
                                                               recompute=recompute)


def prepare_hippo_training_atlases(labels_dir,
                                   result_dir,
                                   image_dir=None,
                                   image_result_dir=None,
                                   smooth=True,
                                   crop_margin=50,
                                   recompute=True):
    """This function prepares training label maps from CobraLab. It first crops each atlas around the right and left
    hippocampi, with a margin. It then equalises the shape of these atlases by croppping them to the size of the
    smallest hippocampus. Finally it realigns the obtained atlases to FS orientation axes.
    :param labels_dir: path of directory with label maps to prepare
    :param result_dir: path of directory where prepared atlases will be writen
    :param image_dir: (optional) path of directory with images corresponding to the label maps to prepare.
    This can be sued to prepare a dataset of real images for supervised training.
    :param image_result_dir: (optional) path of directory where images corresponding to prepared atlases will be writen
    :param smooth: (optional) whether to smooth the final cropped label maps
    :param crop_margin: (optional) margin to add around hippocampi when cropping
    :param recompute: (optional) whether to recompute result files even if they already exists"""

    # create results dir
    utils.mkdir(result_dir)
    tmp_result_dir = os.path.join(result_dir, 'first_cropping')
    utils.mkdir(tmp_result_dir)
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        utils.mkdir(image_result_dir)
        tmp_image_result_dir = os.path.join(image_result_dir, 'first_cropping')
        utils.mkdir(tmp_image_result_dir)
    else:
        tmp_image_result_dir = None

    # list labels and images
    labels_paths = utils.list_images_in_folder(labels_dir)
    if image_dir is not None:
        path_images = utils.list_images_in_folder(image_dir)
    else:
        path_images = [None] * len(labels_paths)

    # crop all atlases around hippo
    print('\ncropping around hippo')
    shape_array = np.zeros((len(labels_paths)*2, 3))
    for idx, (path_label, path_image) in enumerate(zip(labels_paths, path_images)):
        utils.print_loop_info(idx, len(labels_paths), 1)

        # crop left hippo first
        path_label_first_crop_l = os.path.join(tmp_result_dir,
                                               os.path.basename(path_label).replace('.nii', '_left.nii'))
        lab, aff, h = utils.load_volume(path_label, im_only=False)
        lab_l, croppping_idx, aff_l = edit_volumes.crop_volume_around_region(lab, crop_margin,
                                                                             list(range(20101, 20109)), aff=aff)
        if (not os.path.exists(path_label_first_crop_l)) | recompute:
            utils.save_volume(lab_l, aff_l, h, path_label_first_crop_l)
        else:
            lab_l = utils.load_volume(path_label_first_crop_l)
        if path_image is not None:
            path_image_first_crop_l = os.path.join(tmp_image_result_dir,
                                                   os.path.basename(path_image).replace('.nii', '_left.nii'))
            if (not os.path.exists(path_image_first_crop_l)) | recompute:
                im, aff, h = utils.load_volume(path_image, im_only=False)
                im, aff = edit_volumes.crop_volume_with_idx(im, croppping_idx, aff)
                utils.save_volume(im, aff, h, path_image_first_crop_l)
        shape_array[2*idx, :] = np.array(lab_l.shape)

        # crop right hippo and flip them
        path_label_first_crop_r = os.path.join(tmp_result_dir,
                                               os.path.basename(path_label).replace('.nii', '_right_flipped.nii'))
        lab, aff, h = utils.load_volume(path_label, im_only=False)
        lab_r, croppping_idx, aff_r = edit_volumes.crop_volume_around_region(lab, crop_margin,
                                                                             list(range(20001, 20009)), aff=aff)
        if (not os.path.exists(path_label_first_crop_r)) | recompute:
            lab_r = edit_volumes.flip_volume(lab_r, direction='rl', aff=aff_r)
            utils.save_volume(lab_r, aff_r, h, path_label_first_crop_r)
        else:
            lab_r = utils.load_volume(path_label_first_crop_r)
        if path_image is not None:
            path_image_first_crop_r = os.path.join(tmp_image_result_dir,
                                                   os.path.basename(path_image).replace('.nii', '_right.nii'))
            if (not os.path.exists(path_image_first_crop_r)) | recompute:
                im, aff, h = utils.load_volume(path_image, im_only=False)
                im, aff = edit_volumes.crop_volume_with_idx(im, croppping_idx, aff)
                im = edit_volumes.flip_volume(im, direction='rl', aff=aff)
                utils.save_volume(im, aff, h, path_image_first_crop_r)
        shape_array[2*idx+1, :] = np.array(lab_r.shape)

    # list croppped files
    path_labels_first_cropped = utils.list_images_in_folder(tmp_result_dir)
    if tmp_image_result_dir is not None:
        path_images_first_cropped = utils.list_images_in_folder(tmp_image_result_dir)
    else:
        path_images_first_cropped = [None] * len(path_labels_first_cropped)

    # crop all label maps to same size
    print('\nequalising shapes')
    new_shape = np.min(shape_array, axis=0).astype('int32')
    for i, (path_label, path_image) in enumerate(zip(path_labels_first_cropped, path_images_first_cropped)):
        utils.print_loop_info(i, len(path_labels_first_cropped), 1)

        # get cropping indices
        path_lab_cropped = os.path.join(result_dir, os.path.basename(path_label))
        lab, aff, h = utils.load_volume(path_label, im_only=False)
        lab_shape = lab.shape
        min_cropping = np.array([np.maximum(int((lab_shape[i]-new_shape[i])/2), 0) for i in range(3)])
        max_cropping = np.array([min_cropping[i] + new_shape[i] for i in range(3)])

        # crop labels and realign on adni format
        if (not os.path.exists(path_lab_cropped)) | recompute:
            lab, aff = edit_volumes.crop_volume_with_idx(lab, np.concatenate([min_cropping, max_cropping]), aff)
            # realign on adni format
            lab = np.flip(lab, axis=2)
            aff[0:3, 0:3] = np.array([[-0.6, 0, 0], [0, 0, -0.6], [0, -0.6, 0]])
            utils.save_volume(lab, aff, h, path_lab_cropped)

        # crop image and realign on adni format
        if path_image is not None:
            path_im_cropped = os.path.join(image_result_dir, os.path.basename(path_image))
            if (not os.path.exists(path_im_cropped)) | recompute:
                im, aff, h = utils.load_volume(path_image, im_only=False)
                im, aff = edit_volumes.crop_volume_with_idx(im, np.concatenate([min_cropping, max_cropping]), aff)
                im = np.flip(im, axis=2)
                aff[0:3, 0:3] = np.array([[-0.6, 0, 0], [0, 0, -0.6], [0, -0.6, 0]])
                im = edit_volumes.mask_volume(im, lab)
                utils.save_volume(im, aff, h, path_im_cropped)

    # correct all labels to left values
    print('\ncorrecting labels')
    list_incorrect_labels = [77, 80, 251, 252, 253, 254, 255, 29, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54, 58, 60,
                             61, 62, 63, 7012, 20001, 20002, 20004, 20005, 20006, 20007, 20008]
    list_correct_labels = [2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 18, 26, 28, 2, 30, 31, 20108,
                           20101, 20102, 20104, 20105, 20106, 20107, 20108]
    edit_volumes.correct_labels_in_dir(result_dir, list_incorrect_labels, list_correct_labels, result_dir)

    # smooth labels
    if smooth:
        print('\nsmoothing labels')
        edit_volumes.smooth_labels_in_dir(result_dir, result_dir)


def prepare_hippo_testing_images(main_image_dir,
                                 main_result_dir,
                                 target_res,
                                 padding_margin=85,
                                 delete_intermediate_files=True,
                                 path_freesurfer='/usr/local/freesurfer/',
                                 verbose=True,
                                 recompute=True):
    """This function creates multi-modal images of the right and left hippocampi at the target resolution.
    In that purpose it loops over subjects (assumed to be sorted between healthy and AD subfolders) and calls
    preprocess_adni_hippo on each of them.
    :param main_image_dir: path of main directory with images to prepare for testing. Should be organised as follows:
    main_image_dir/state_dir(AD or healthy)/subject_dir/images(t1.mgz, t2.mgz, and aseg.mgz)
    :param main_result_dir: path of main directory where prepared images and labels will be writen.
    Will be organised as follows: main_result_dir/state_dir(AD or healthy)/subject_dir/images(hippo_left.nii.gz,
    hippo_right.nii.gz, hippo_left_aseg.nii.gz, hippo_right_aseg.nii.gz)
    :param target_res: resolution at which to resample the label maps, and the images.
    Can be a number (isotropic resolution), a sequence, or a 1d numpy array.
    :param padding_margin: (optional) margin to add around hippocampi when cropping
    :param delete_intermediate_files: (optional) whether to delete temporary files. Default is True.
    :param path_freesurfer: (optional) path of FreeSurfer home, to use mri_convert
    :param verbose: (optional) whether to print out mri_convert output when resampling images.
    :param recompute: (optional) whether to recompute result files even if they already exists"""

    # create results dir
    utils.mkdir(main_result_dir)

    # loop over states (i.e. AD and healthy)
    list_states = utils.list_subfolders(main_image_dir, whole_path=False)
    for state in list_states:

        # create state directory in result folder
        state_dir = os.path.join(main_image_dir, state)
        result_state_dir = os.path.join(main_result_dir, state)
        utils.mkdir(result_state_dir)

        # loop over subjects
        list_subjects = utils.list_subfolders(state_dir, whole_path=False)
        for subject in list_subjects:

            # create subject directoty in state subfolder
            subject_dir = os.path.join(state_dir, subject)
            result_subject_dir = os.path.join(result_state_dir, subject)
            utils.mkdir(result_subject_dir)

            # get file paths
            t1_path = os.path.join(subject_dir, 't1.mgz')
            t2_path = os.path.join(subject_dir, 't2.mgz')
            aseg_path = os.path.join(subject_dir, 'aseg.mgz')

            preprocess_adni_hippo(t1_path,
                                  t2_path,
                                  aseg_path,
                                  result_subject_dir,
                                  target_res,
                                  padding_margin,
                                  remove=delete_intermediate_files,
                                  path_freesurfer=path_freesurfer,
                                  verbose=verbose,
                                  recompute=recompute)


def preprocess_adni_hippo(path_t1,
                          path_t2,
                          path_aseg,
                          result_dir,
                          target_res,
                          padding_margin=85,
                          remove=False,
                          path_freesurfer='/usr/local/freesurfer/',
                          verbose=True,
                          recompute=True):
    """This function builds a T1+T2 multimodal image from the ADNI dataset.
    It first rescales intensities of each channel between 0 and 255.
    It then resamples the T2 image (which are 0.4*0.4*2.0 resolution) to target resolution.
    The obtained T2 is then padded in all directions by the padding_margin param (typically large 85).
    The T1 and aseg are then resampled like the T2 using mri_convert.
    Now that the T1, T2 and asegs are aligned and at the same resolution, we crop them around the right and left hippo.
    Finally, the T1 and T2 are concatenated into one single multimodal image.
    :param path_t1: path input T1 (typically at 1mm isotropic)
    :param path_t2: path input T2 (typically cropped around the hippo in sagittal axis, 0.4x0.4x2.0)
    :param path_aseg: path input segmentation (typically at 1mm isotropic)
    :param result_dir: path of directory where prepared images and labels will be writen.
    :param target_res: resolution at which to resample the label maps, and the images.
    Can be a number (isotropic resolution), a sequence, or a 1d numpy array.
    :param padding_margin: (optional) margin to add around hippocampi when cropping
    :param remove: (optional) whether to delete temporary files. Default is True.
    :param path_freesurfer: (optional) path of FreeSurfer home, to use mri_convert
    :param verbose: (optional) whether to print out mri_convert output when resampling images
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create results dir
    utils.mkdir(result_dir)

    path_test_im_right = os.path.join(result_dir, 'hippo_right.nii.gz')
    path_test_aseg_right = os.path.join(result_dir, 'hippo_right_aseg.nii.gz')
    path_test_im_left = os.path.join(result_dir, 'hippo_left.nii.gz')
    path_test_aseg_left = os.path.join(result_dir, 'hippo_left_aseg.nii.gz')
    if (not os.path.isfile(path_test_im_right)) | (not os.path.isfile(path_test_aseg_right)) | \
       (not os.path.isfile(path_test_im_left)) | (not os.path.isfile(path_test_aseg_left)) | recompute:

        # set up FreeSurfer
        os.environ['FREESURFER_HOME'] = path_freesurfer
        os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
        mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin')

        # rescale T1
        path_t1_rescaled = os.path.join(result_dir, 't1_rescaled.nii.gz')
        if (not os.path.isfile(path_t1_rescaled)) | recompute:
            im, aff, h = utils.load_volume(path_t1, im_only=False)
            im = edit_volumes.rescale_volume(im)
            utils.save_volume(im, aff, h, path_t1_rescaled)
        # rescale T2
        path_t2_rescaled = os.path.join(result_dir, 't2_rescaled.nii.gz')
        if (not os.path.isfile(path_t2_rescaled)) | recompute:
            im, aff, h = utils.load_volume(path_t2, im_only=False)
            im = edit_volumes.rescale_volume(im)
            utils.save_volume(im, aff, h, path_t2_rescaled)

        # resample T2 to target res
        path_t2_resampled = os.path.join(result_dir, 't2_rescaled_resampled.nii.gz')
        if (not os.path.isfile(path_t2_resampled)) | recompute:
            str_res = ' '.join([str(r) for r in utils.reformat_to_list(target_res, length=3)])
            cmd = mri_convert + ' ' + path_t2_rescaled + ' ' + path_t2_resampled + ' --voxsize ' + str_res
            cmd += ' -odt float'
            if not verbose:
                cmd += ' >/dev/null 2>&1'
            _ = os.system(cmd)

        # pad T2
        path_t2_padded = os.path.join(result_dir, 't2_rescaled_resampled_padded.nii.gz')
        if (not os.path.isfile(path_t2_padded)) | recompute:
            t2, aff, h = utils.load_volume(path_t2_resampled, im_only=False)
            t2_padded = np.pad(t2, padding_margin, 'constant')
            aff[:3, -1] = aff[:3, -1] - (aff[:3, :3] @ (padding_margin * np.ones((3, 1)))).T
            utils.save_volume(t2_padded, aff, h, path_t2_padded)

        # resample T1 and aseg accordingly
        path_t1_resampled = os.path.join(result_dir, 't1_rescaled_resampled.nii.gz')
        if (not os.path.isfile(path_t1_resampled)) | recompute:
            cmd = mri_convert + ' ' + path_t1_rescaled + ' ' + path_t1_resampled + ' -rl ' + path_t2_padded
            cmd += ' -odt float'
            if not verbose:
                cmd += ' >/dev/null 2>&1'
            _ = os.system(cmd)
        path_aseg_resampled = os.path.join(result_dir, 'aseg_resampled.nii.gz')
        if (not os.path.isfile(path_aseg_resampled)) | recompute:
            cmd = mri_convert + ' ' + path_aseg + ' ' + path_aseg_resampled + ' -rl ' + path_t2_padded
            cmd += ' -rt nearest -odt float'
            if not verbose:
                cmd += ' >/dev/null 2>&1'
            _ = os.system(cmd)

        # crop images and concatenate T1 and T2
        for lab, side in zip([17, 53], ['left', 'right']):
            path_test_image = os.path.join(result_dir, 'hippo_{}.nii.gz'.format(side))
            path_test_aseg = os.path.join(result_dir, 'hippo_{}_aseg.nii.gz'.format(side))
            if (not os.path.isfile(path_test_image)) | (not os.path.isfile(path_test_aseg)) | recompute:
                aseg, aff, h = utils.load_volume(path_aseg_resampled, im_only=False)
                tmp_aseg, cropping, tmp_aff = edit_volumes.crop_volume_around_region(aseg,
                                                                                     margin=30,
                                                                                     masking_labels=lab,
                                                                                     aff=aff)
                if side == 'right':
                    tmp_aseg = edit_volumes.flip_volume(tmp_aseg, direction='rl', aff=tmp_aff)
                utils.save_volume(tmp_aseg, tmp_aff, h, path_test_aseg)
                if (not os.path.isfile(path_test_image)) | recompute:
                    t1 = utils.load_volume(path_t1_resampled)
                    t1 = edit_volumes.crop_volume_with_idx(t1, crop_idx=cropping)
                    t1 = edit_volumes.mask_volume(t1, tmp_aseg, dilate=6, erode=5)
                    t2 = utils.load_volume(path_t2_padded)
                    t2 = edit_volumes.crop_volume_with_idx(t2, crop_idx=cropping)
                    t2 = edit_volumes.mask_volume(t2, tmp_aseg, dilate=6, erode=5)
                    if side == 'right':
                        t1 = edit_volumes.flip_volume(t1, direction='rl', aff=tmp_aff)
                        t2 = edit_volumes.flip_volume(t2, direction='rl', aff=tmp_aff)
                    test_image = np.stack([t1, t2], axis=-1)
                    utils.save_volume(test_image, tmp_aff, h, path_test_image)

        # remove unnecessary files
        if remove:
            list_files_to_remove = [path_t1_rescaled,
                                    path_t2_rescaled,
                                    path_t2_resampled,
                                    path_t2_padded,
                                    path_t1_resampled,
                                    path_aseg_resampled]
            for path in list_files_to_remove:
                os.remove(path)


if __name__ == '__main__':

    # --------------------------- downsample/resample Buckner40 images T1-spacing experiment ---------------------------

    # image and labels folders
    image_folder = '/home/benjamin/data/mit/Buckner40/images/orig_training'
    lab_folder = '/home/benjamin/data/mit/Buckner40/labels/training/extra_cerebral_generation_rl_regrouped'
    # result folders
    im_results_folder = '/home/benjamin/data/mit/Buckner40/images/orig_training_resampled'
    downsample_result_dir = '/home/benjamin/data/mit/Buckner40/images/orig_training_downsampled'
    lab_result_folder = '/home/benjamin/data/mit/Buckner40/labels/training/' \
                        'extra_cerebral_segmentation_rl_regrouped_downsampled'
    # downsample/resample images
    prepare_anisotropic_dataset(image_dir=image_folder,
                                list_synth_res=[3, 6, 9],
                                downsample_image_result_dir=downsample_result_dir,
                                resample_image_result_dir=im_results_folder,
                                labels_dir=lab_folder,
                                downsample_labels_result_dir=lab_result_folder,
                                recompute=False)

    # ----------------------------- preprocess atlases for hippocampal subfields experiment ----------------------------

    labels_folder = '/data/CobraLab/hippo/full_labels'
    result_folder = '/data/CobraLab/hippo/atlases_labels'
    image_folder = '/data/CobraLab/hippo/full_images'
    image_result_folder = '/data/CobraLab/hippo/atlases_images'
    prepare_hippo_training_atlases(labels_folder,
                                   result_folder,
                                   image_dir=image_folder,
                                   image_result_dir=image_result_folder,
                                   smooth=True,
                                   crop_margin=35,
                                   recompute=True)

    # ------------------------------ preprocess ADNI scans of the hippocampus for testing ------------------------------

    main_folder = '/data/ADNI/subjects_t1_t2_aseg/'
    main_result_folder = '/data/ADNI/subjects_hippos_0.4'
    prepare_hippo_testing_images(main_folder,
                                 main_result_folder,
                                 target_res=0.4,
                                 padding_margin=85,
                                 delete_intermediate_files=True,
                                 path_freesurfer='/usr/local/freesurfer/',
                                 verbose=True,
                                 recompute=True)
