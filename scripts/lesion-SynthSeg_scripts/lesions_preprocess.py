# python imports
import os
import numpy as np

# project imports
from SynthSeg.boxplots import draw_boxplots
from SynthSeg.evaluate import dice_evaluation

# third-party imports
from ext.lab2im.edit_volumes import correct_labels_in_dir, mri_convert_images_in_dir


incorrect_labels = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/samseg_incorrect_labels_generation.npy'
correct_labels = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/samseg_correct_labels_generation.npy'
segmentation_labels = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/segmentation_labels.npy'

gt_dir = '/home/benjamin/data/lesions/MS/labels/resample_1_1_1/samseg_lesions_cropped_corrected'
gt_dir_testing = '/home/benjamin/data/lesions/MS/labels/resample_1_1_1/samseg_lesions_cropped_corrected_testing'

samseg_dirs = ['/home/benjamin/data/lesions/MS/samseg_lesions/t1',
               '/home/benjamin/data/lesions/MS/samseg_lesions/flair',
               '/home/benjamin/data/lesions/MS/samseg_lesions/t1_flair']

pvsynthseg_dirs = ['/home/benjamin/data/lesions/MS/PV-SynthSeg/models/t1/segmentation_070',
                   '/home/benjamin/data/lesions/MS/PV-SynthSeg/models/flair/segmentation_070',
                   '/home/benjamin/data/lesions/MS/PV-SynthSeg/models/t1_flair/segmentation_070']

my_palette = {'SAMSEG all ROIs': (0.9137254901960784, 0.3686274509803921, 0.050980392156862744),
              'SAMSEG lesions only': (0.9921568627450981, 0.5793310265282584, 0.272879661668589),
              'PV-SynthSeg all ROIs': (0.11787773933102653, 0.4283737024221453, 0.6982698961937717),
              'PV-SynthSeg lesions only': (0.38682045367166473, 0.6599615532487505, 0.8264206074586697)}
orders = ['t1', 'flair', 't1_flair']

names = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/eval_names.npy'
eval_indices = '/home/benjamin/data/lesions/MS/PV-SynthSeg/labels_classes_stats/eval_indices.npy'

list_dice = list()
list_datasets = list()
list_methods = list()

for samseg_dir in samseg_dirs:
    samseg_corrected_dir = samseg_dir + '_corrected'
    samseg_aligned_dir = samseg_corrected_dir + '_resampled_111_cropped'
    samseg_aligned_dir_testing = samseg_corrected_dir + '_resampled_111_cropped_testing'
    path_dice = os.path.join(samseg_aligned_dir, 'dice.npy')
    path_dice_lesions = os.path.join(samseg_aligned_dir, 'dice_lesions.npy')
    path_dice_testing = os.path.join(samseg_aligned_dir_testing, 'dice.npy')
    path_dice_lesions_testing = os.path.join(samseg_aligned_dir_testing, 'dice_lesions.npy')
    # correct_labels_in_dir(samseg_dir,
    #                       incorrect_labels,
    #                       correct_labels,
    #                       samseg_corrected_dir,
    #                       recompute=False)
    # mri_convert_images_in_dir(samseg_corrected_dir,
    #                           samseg_aligned_dir,
    #                           interpolation='nearest',
    #                           reference_dir=gt_dir,
    #                           recompute=False)
    # dice = dice_evaluation(gt_dir, samseg_aligned_dir, segmentation_labels, path_dice)
    # np.save(path_dice_lesions, dice[4, :])
    # dice = dice_evaluation(gt_dir_testing, samseg_aligned_dir_testing, segmentation_labels, path_dice_testing)
    # np.save(path_dice_lesions_testing, dice[4, :])
    list_dice.append(path_dice_testing)
    list_dice.append(path_dice_lesions_testing)
    list_datasets.append(os.path.basename(samseg_dir))
    list_datasets.append(os.path.basename(samseg_dir))
    list_methods.append('SAMSEG all ROIs')
    list_methods.append('SAMSEG lesions only')

for pvsynthseg_dir in pvsynthseg_dirs:
    path_dice_testing = os.path.join(pvsynthseg_dir, 'dice.npy')
    path_dice_lesions_testing = os.path.join(pvsynthseg_dir, 'dice_lesions.npy')
    # dice = dice_evaluation(gt_dir_testing, pvsynthseg_dir, segmentation_labels, path_dice_testing)
    # np.save(path_dice_lesions_testing, dice[4, :])
    list_dice.append(path_dice_testing)
    list_dice.append(path_dice_lesions_testing)
    list_datasets.append(os.path.basename(os.path.dirname(pvsynthseg_dir)))
    list_datasets.append(os.path.basename(os.path.dirname(pvsynthseg_dir)))
    list_methods.append('PV-SynthSeg all ROIs')
    list_methods.append('PV-SynthSeg lesions only')

draw_boxplots(list_dice,
              list_datasets,
              list_methods,
              names,
              eval_indices,
              palette=None,
              order=None,
              figsize=None,
              remove_legend=False,
              fontsize=23,
              fontsize_legend=16,
              y_label=None,
              boxplot_width=0.6,
              boxplot_linewidth=2,
              outlier_size=4,
              av_plot_ylim=[0, 1],
              av_plot_step_yticks=0.1,
              av_plot_title='Average Dice across subjects')
