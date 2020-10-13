"""Example to show how to estimate of the hyperparameters governing the GMM prior distributions.
We do not provide example images, so do not try to run this directly.
Here we show how to estimate the hyperparameters from images of two modalities:
T1-weighted and T2-weighted (respectively in /data/t1, and /data/t2).
We suppose that the data in each folder correspond to the same subjects.
As such we can use the same label maps for the two modalities: that's why we only specify one folder in labels_dir."""

from SynthSeg.estimate_priors import build_intensity_stats

list_image_dir = ['/data/t1', '/data/t2']
labels_dir = '/data/labels'
estimation_labels = '../data_example/generation_labels.npy'
estimation_classes = '../data_example/generation_classes.npy'
result_dir = './'

build_intensity_stats(list_image_dir=list_image_dir,
                      list_labels_dir=labels_dir,
                      estimation_labels=estimation_labels,
                      estimation_classes=estimation_classes,
                      result_dir=result_dir,
                      rescale=True)
