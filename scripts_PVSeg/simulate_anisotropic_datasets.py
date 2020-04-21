# python imports
import os
import logging
import numpy as np

# third-party imports
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
    if not os.path.isdir(resample_image_result_dir):
        os.mkdir(resample_image_result_dir)
    if not os.path.isdir(downsample_image_result_dir):
        os.mkdir(downsample_image_result_dir)
    if downsample_labels_result_dir is not None:
        if not os.path.isdir(downsample_labels_result_dir):
            os.mkdir(downsample_labels_result_dir)

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


if __name__ == '__main__':
    logging.getLogger('tensorflow').disabled = True

    # image and labels folders
    image_folder = '/home/benjamin/data/mit/Buckner40/images/orig_training'
    lab_folder = '/home/benjamin/data/mit/Buckner40/labels/training/extra_cerebral_generation_rl_regrouped'

    # result folders
    im_results_folder = '/home/benjamin/data/mit/Buckner40/images/orig_training_resampled'
    downsample_result_dir = '/home/benjamin/data/mit/Buckner40/images/orig_training_downsampled'
    lab_result_folder = '/home/benjamin/data/mit/Buckner40/labels/training/' \
                        'extra_cerebral_segmentation_rl_regrouped_downsampled'

    prepare_anisotropic_dataset(image_dir=image_folder,
                                list_synth_res=[3, 6, 9],
                                downsample_image_result_dir=downsample_result_dir,
                                resample_image_result_dir=im_results_folder,
                                labels_dir=lab_folder,
                                downsample_labels_result_dir=lab_result_folder,
                                recompute=False)
