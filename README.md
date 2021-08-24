# SynthSeg

In this repository, we present SynthSeg, the first convolutional neural network to readily segment brain MRI scans of
any contrast and resolution, with an output that is 1 mm isotropic, no matter what the resolution of the input is. 
Additionally, the proposed model is robust to:
- a wide array of subject populations: from young and healthy to ageing and diseased subjects with prominent atrophy,
- white matter lesions,
- and scans with or without preprocessing (bias field corruption, skull stripping, intensity normalisation, registration to
template).

As a result, SynthSeg only relies on a single model, which we distribute here. We emphasise that this model can be used
out-of-the-box without retraining or fine-tuning, and can run on the GPU (6s per scan) as well as the CPU (1min).
\
\
![Generation examples](data/README_figures/segmentations.png)

----------------

### Easily segment your data with one command

Once all the python packages are installed (see below), you can simply test SynthSeg on your own data with:
```
python ./scripts/commands/SynthSeg_predict.py <images> <segmentations> --post <post> --vol <vol> --resample <resample>
```
where:
- `<image>` is the path to an image to segment. \
This can also be a folder, in which case all the image inside that folder will be segmented.
- `<segmentation>` is the path where the output segmentation will be saved. \
This must be a folder if `<image>` designates a folder.
- `<post>` (optional) is the path where the posteriors (given as soft probability maps) will be saved. \
This must be a folder if `<image>` designates a folder.
- `<vol>` (optional) is the path to an output csv file where the volumes of every segmented structures
will be saved for all scans (i.e., one csv file for all subjects; e.g. /path/to/volumes.csv)
- `<resample>` (optional) SynthSeg segmentations are always given at 1mm isotropic resolution. Therefore, 
images are internally resampled to this resolution (except if they aleady are at 1mm resolution). 
Use this optional flag to save the resampled images: it must be the path to a single image, or a folder
if `<image>` designates a folder.

\
Additional optional flags are also available:
- `--cpu`: to enforce the code to run on the CPU, even if a GPU is available.
- `--threads`: to indicate the number of cores to be used if running on a CPU (example: `--threads 3` to run on 3 cores).
 This value defaults to 1, but we recommend increasing it for faster analysis.
- `--crop`: to crop the input images to a given shape before segmentation (example: `--crop 160` to run on 
160<sup>3</sup> patches). Images are cropped around their centre, and their segmentations are given in native space 
(i.e., at the original size). Use this flag for faster analysis or if you have a GPU with insufficient memory
to process the whole image.


**IMPORTANT:** Because SynthSeg may produce segmentations at higher resolution than the images (i.e., at 
1mm<sup>3</sup>), some viewers will not display them correctly when overlaying the segmentations on the
original images. If that’s the case, you can use the `--resample` flag to obtain a resampled image that
lives in the same space as the segmentation, such that any viewer can be used to visualize them together.
We highlight that the resampling is performed internally to avoid the dependence on any external tool.

The complete list of segmented structures is available in [labels table.txt](data/labels%20table.txt) along with their
corresponding values. This table also details the order in which the posteriors maps are sorted.

----------------

### Requirements

All the python requirements are listed in requirements.txt. We list here the important dependencies:

- tensorflow-gpu 2.0.2
- keras 2.3.1
- nibabel
- numpy, scipy, sklearn, tqdm, pillow, matplotlib, ipython, ...

This code also relies on several external packages (already included in `\ext` for convenience):

- [lab2im](https://github.com/BBillot/lab2im): contains functions for data augmentation, and a simple version of 
 the generative model, on which we build to build `label_to_image_model`
- [neuron](https://github.com/adalca/neuron): contains functions for deforming, and resizing tensors, as well as 
functions to build the segmentation network [1,2].
- [pytool-lib](https://github.com/adalca/pytools-lib): library required by the *neuron* package.

If you wish to run SynthSeg on the GPU, or to train your own model, you will also need the usual deep learning libraries:
- Cuda 10.0
- CUDNN 7.0


----------------

### How does it work ?

In short, we train a network with synthetic images sampled on the fly from a generative model based on the forward
model of Bayesian segmentation. Crucially, we adopt a domain randomisation strategy where we fully randomise the 
generation parameters which are drawn from uninformative uniform distributions. Therefore, by maximising the variability
of the training data, we force to learn domain-agnostic features. As a result SynthSeg is able to readily segment
real scans of any target domain, without retraining or fine-tuning. 

The following figure illustrates the the workflow of a training iteration, and provides an overview of the generative 
model:
\
\
![Generation examples](data/README_figures/overview.png)
\
\
Finally we show additional examples of the synthesised images along with an overlay of their target segmentations:
\
\
![Generation examples](data/README_figures/training_data.png)
\
\
If you are interested to learn more about SynthSeg, you can read the associated publication (see below), and watch this
presentation, which was given at MIDL 2020 for a related article on a preliminary version of SynthSeg (robustness to
MR contrast but not resolution).
\
\
[![Talk SynthSeg](data/README_figures/youtube_link.png)](https://www.youtube.com/watch?v=Bfp3cILSKZg&t=1s)

----------------

### Train your own model

This repository contains all the code and data necessary to train your own network. Importantly, the proposed method
only requires a set of anatomical segmentations to be trained which we include in [data](data/training_label_maps).
Regarding the code, we include functions to [train](SynthSeg/training.py) new models, as well as to 
[validate](SynthSeg/validate.py), and [test](SynthSeg/predict.py) them. While these functions are thoroughly documented,
you can familiarise yourself with the different aspects of the code by following the provided tutorials:

- [1-generation_visualisation](scripts/tutorials/1-generation_visualisation.py): We recommend you to start here, as this
very simple script shows examples of the synthetic images used to train SynthSeg.

- [2-generation_explained](scripts/tutorials/2-generation_explained.py): This second script describes all the parameters
used to control the generative model that we sample from during training. We advise you to thoroughly read this 
tutorial, as it is essential to undesrtand how the synthetic data is formed before starting training.

- [3-training](scripts/tutorials/3-training.py): This scripts reuses teh parameters explained in the previous tutorial
and focuses on the learning/architecture parameters. The script here is the very one we used to train SynthSeg !

- [4-generation_advanced](scripts/tutorials/4-generation_advanced.py): Here we detail more advanced generation options, 
in the case of training a version of SynthSeg that is specific to a given contrast and/or resolution, like we did for
the SynthSeg variants (although they were shown to be outperformed by the broader SynthSeg model trained in the 3rd
tutorial).

- [5-intensity_estimation](scripts/tutorials/5-intensity_estimation.py): Finally, this script shows how to estimate the 
Gaussian priors of the GMM when training a contrast-specific version of SynthSeg.

These tutorials cover a lot of materials and will enable you to train your own SynthSeg model. Moreover, if you wish,
you can have access to more detialed information by reading the docstrings of all functions, which contain very detailed
information.

----------------

### Content

- [SynthSeg](SynthSeg): this is the main folder containing the generative model and training function:

  - [labels_to_image_model.py](SynthSeg/labels_to_image_model.py): contains the generative model `labels_to_image_model`.
  
  - [brain_generator.py](SynthSeg/brain_generator.py): contains the class `BrainGenerator`, which is a wrapper around 
  `labels_to_image_model`. New images can simply be generated by instantiating an object of this class, and call the 
  method `generate_image()`.
  
  - [training.py](SynthSeg/training.py): contains the function `training` to train the segmentation network. This function
  provides an example of how to integrate the labels_to_image_model in a broader GPU model. All training parameters are 
  explained there.
  
  - [predict.py](SynthSeg/predict.py): function to predict segmentations of new MRI scans. Can also be used for testing.
   
  - [validate.py](SynthSeg/validate.py): contains `validate_training` to validate the models saved during training, as 
  validation on real images has to be done offline.
 
 - [models](models): this is where you will find the trained model for SynthSeg.
 
- [data](data): this folder contains some examples of brain label maps if you wish to train your own SynthSeg model.
 
- [script](scripts): additionally to the tutorials, we also provide functions to launch trainings and testings from the 
terminal.

- [ext](ext): contains external packages, especially the *lab2im* package, and a modified version of *neuron*.


----------------

### Citation/Contact

If you use this code, please cite the following papers:

**SynthSeg: Domain Randomisation for Segmentation of Brain MRI Scans of any Contrast and Resolution** \
Benjamin Billot, Douglas N. Greve, Oula Puonti, Axel Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V. Dalca, Juan Eugenio Iglesias \
[[arxiv](https://arxiv.org/abs/2107.09559)]

**A Learning Strategy for Contrast-agnostic MRI Segmentation** \
Benjamin Billot, Douglas N. Greve, Koen Van Leemput, Bruce Fischl, Juan Eugenio Iglesias*, Adrian V. Dalca* \
*contributed equally \
MIDL 2020 \
[[link](http://proceedings.mlr.press/v121/billot20a.html) | [arxiv](https://arxiv.org/abs/2003.01995) | [bibtex](bibtex.txt)]

**Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast** \
Benjamin Billot, Eleanor D. Robinson, Adrian V. Dalca, Juan Eugenio Iglesias \
MICCAI 2020 \
[[link](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18) | [arxiv](https://arxiv.org/abs/2004.10221) | [bibtex](bibtex.txt)]

If you have any question regarding the usage of this code, or any suggestions to improve it you can contact us at: \
benjamin.billot.18@ucl.ac.uk


----------------

### References

[1] *[Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation](http://www.mit.edu/~adalca/files/papers/cvpr2018_priors.pdf)* \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
CVPR 2018

[2] *[Unsupervised Data Imputation via Variational Inference of Deep Subspaces](https://arxiv.org/abs/1903.03503)* \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
Arxiv preprint 2019
