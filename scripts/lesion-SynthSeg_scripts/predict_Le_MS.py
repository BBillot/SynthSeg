from SynthSeg.predict import predict

predict('/home/benjamin/data/lesions/MS/images/resampled_1_6.5_1_upsampled_1_1_1/flair',
        '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/models_MS_contrast/models/dice_014.h5',
        '/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/labels_classes_stats/segmentation_labels.npy',
        path_segmentations='/home/benjamin/data/lesions/Le/PV-SynthSeg_VAE/models_MS_contrast/segmentation_14_epochs_flair',
        path_posteriors=None,
        path_volumes=None,
        skip_background_volume=True,
        padding=None,
        cropping=None,
        resample=None,
        sigma_smoothing=0,
        keep_biggest_component=False,
        gt_folder=None)
