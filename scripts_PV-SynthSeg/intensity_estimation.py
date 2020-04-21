"""estimation of the hyperparameters governing the GMM prior distribution
here we estimate the hyperparameters for two modalities t1 and t2
because the different modalities are estimated on the same subjects, we can use the same label maps."""

from SynthSeg.estimate_priors import build_intensity_stats_for_several_modalities

list_image_dir = ['/data/t1', '/data/t2']
labels_dir = '/data/labels'
estimation_labels = './labels_classes_stats/sampling_labels.npy'
estimation_classes = './labels_classes_stats/sampling_classes.npy'
generation_classes = './labels_classes_stats/cobralab_atlases_classes.npy'
result_dir = './labels_classes_stats'

build_intensity_stats_for_several_modalities(list_image_dir=list_image_dir,
                                             list_labels_dir=labels_dir,
                                             estimation_labels=estimation_labels,
                                             estimation_classes=estimation_classes,
                                             generation_classes=generation_classes,
                                             results_dir=result_dir,
                                             rescale=True)
