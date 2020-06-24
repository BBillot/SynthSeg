import os
from SynthSeg.validate import validate_training, plot_validation_curves

# MS
label_list = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/segmentation_labels.npy'
gt_folder = '/home/benjamin/data/lesions/MS/labels/resampled_1_6.5_1_upsampled_1_1_1/samseg_lesions'
list_image_folder = ['/home/benjamin/data/lesions/MS/images/resampled_1_6.5_1_upsampled_1_1_1/flair',
                     '/home/benjamin/data/lesions/MS/images/resampled_1_6.5_1_upsampled_1_1_1/t1']
main_folder = '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/models_MS_contrast'

list_net_validation_dirs = list()
for image_folder in list_image_folder:

    models_folder = os.path.join(main_folder, 'models')
    validation_main_folder = os.path.join(main_folder, 'validation_' + os.path.basename(image_folder))
    validate_training(image_dir=image_folder,
                      gt_dir=gt_folder,
                      models_dir=models_folder,
                      validation_main_dir=validation_main_folder,
                      path_label_list=label_list, recompute=True)
    list_net_validation_dirs.append(validation_main_folder)

plot_validation_curves(list_net_validation_dirs)