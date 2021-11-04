# SynthSeg

\
\
:tada: Update 29/10/2021: SynthSeg is now available on the dev version of 
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)   !! :tada: \
See [here](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) on how to use it
\
\
\
In this repository, we present SynthSeg, the first convolutional neural network to readily segment brain MRI scans of
any contrast and resolution, with an predictions at 1mm isotropic resolution, regardless of the input resolution. 
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
python ./scripts/commands/SynthSeg_predict.py --i <image> --o <segmentation> --post <post> --resample <resample> --vol <vol>
```
where:
- `<image>` is the path to an image to segment (supported formats are .nii, .nii.gz, and .mgz). \
This can also be a folder, in which case all the image inside that folder will be segmented.
- `<segmentation>` is the path where the output segmentation(s) will be saved. \
This must be a folder if `<image>` designates a folder.
- `<post>` (optional) is the path where the posteriors (given as soft probability maps) will be saved. \
This must be a folder if `<image>` designates a folder.
- `<resample>` (optional) SynthSeg segmentations are always given at 1mm isotropic resolution. Therefore, 
images are internally resampled to this resolution (except if they aleady are at 1mm resolution). 
Use this optional flag to save the resampled images: it must be the path to a single image, or a folder
if `<image>` designates a folder.
- `<vol>` (optional) is the path to an output csv file where the volumes of every segmented structures
will be saved for all scans (i.e., one csv file for all subjects; e.g. /path/to/volumes.csv)

\
Additional optional flags are also available:
- `--cpu`: to enforce the code to run on the CPU, even if a GPU is available.
- `--threads`: to indicate the number of cores to be used if running on a CPU (example: `--threads 3` to run on 3 cores).
This value defaults to 1, but we recommend increasing it for faster analysis.
- `--crop`: to crop the input images to a given shape before segmentation. The given size must be divisible by 32.
Images are cropped around their centre, and their segmentations are given at the original size). It can be given as a 
single (i.e., `--crop 160` to run on 160<sup>3</sup> patches), or several integers (i.e, `--crop 160 128 192` to crop to
different sizes in each direction, ordered in RAS coordinates). This value defaults to 192, but it can be decreased
for faster analysis or to fit in your GPU.


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

- Python 3.6 (this is important to have access to the right keras and tensorflow versions!)
- tensorflow-gpu 2.0.1
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

This repository contains all the code and data necessary to train, validate, and test your own network. Importantly, the
proposed method only requires a set of anatomical segmentations to be trained (no images), which we include in 
[data](data/training_label_maps). While the provided functions are thoroughly documented, we highly recommend to start 
with the following tutorials:

- [1-generation_visualisation](scripts/tutorials/1-generation_visualisation.py): This very simple script shows examples
of the synthetic images used to train SynthSeg.

- [2-generation_explained](scripts/tutorials/2-generation_explained.py): This second script describes all the parameters
used to control the generative model. We advise you to thoroughly follow this tutorial, as it is essential to understand
how the synthetic data is formed before you start training your own models.

- [3-training](scripts/tutorials/3-training.py): This scripts re-uses the parameters explained in the previous tutorial
and focuses on the learning/architecture parameters. The script here is the very one we used to train SynthSeg !

- [4-training](scripts/tutorials/4-prediction.py): This scripts shows how to make predictions, once the network has been 
trained.

- [5-generation_advanced](scripts/tutorials/5-generation_advanced.py): Here we detail more advanced generation options, 
in the case of training a version of SynthSeg that is specific to a given contrast and/or resolution (although these
types of variants were shown to be outperformed by the SynthSeg model trained in the 3rd tutorial).

- [6-intensity_estimation](scripts/tutorials/6-intensity_estimation.py): Finally, this script shows how to estimate the 
Gaussian priors of the GMM when training a contrast-specific version of SynthSeg.

These tutorials cover a lot of materials and will enable you to train your own SynthSeg model. Moreover, even more 
detailed information is provided in the docstrings of all functions, so don't hesitate to have a look at these !

----------------

### Content

- [SynthSeg](SynthSeg): this is the main folder containing the generative model and training function:

  - [labels_to_image_model.py](SynthSeg/labels_to_image_model.py): contains the generative model for MRI scans.
  
  - [brain_generator.py](SynthSeg/brain_generator.py): contains the class `BrainGenerator`, which is a wrapper around 
  `labels_to_image_model`. New images can simply be generated by instantiating an object of this class, and call the 
  method `generate_image()`.
  
  - [training.py](SynthSeg/training.py): contains code to train the segmentation network (with explainations for all 
  training parameters). This function also shows how to integrate the generative model in a training setting.
  
  - [predict.py](SynthSeg/predict.py): prediction and testing.
   
  - [validate.py](SynthSeg/validate.py): includes code for validation (which has to be done offline on real images).
 
- [models](models): this is where you will find the trained model for SynthSeg.
 
- [data](data): this folder contains some examples of brain label maps if you wish to train your own SynthSeg model.
 
- [script](scripts): contains tutorials as well as scripts to launch trainings and testings from a terminal.

- [ext](ext): includes external packages, especially the *lab2im* package, and a modified version of *neuron*.


----------------

### Citation/Contact

This code is under [Apache 2.0](LICENSE.txt) licensing. \
If you use it, please cite one of the following papers:

**SynthSeg: Domain Randomisation for Segmentation of Brain MRI Scans of any Contrast and Resolution** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
[[arxiv](https://arxiv.org/abs/2107.09559) | [bibtex](bibtex.bib)]

**A Learning Strategy for Contrast-agnostic MRI Segmentation** \
B. Billot, D.N. Greve, K. Van Leemput, B. Fischl, J.E. Iglesias*, A.V. Dalca* \
*contributed equally \
MIDL 2020 \
[[link](http://proceedings.mlr.press/v121/billot20a.html) | [arxiv](https://arxiv.org/abs/2003.01995) | [bibtex](bibtex.bib)]

**Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast** \
B. Billot, E.D. Robinson, A.V. Dalca, J.E. Iglesias \
MICCAI 2020 \
[[link](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18) | [arxiv](https://arxiv.org/abs/2004.10221) | [bibtex](bibtex.bib)]

If you have any question regarding the usage of this code, or any suggestions to improve it, you can contact us at: \
benjamin.billot.18@ucl.ac.uk


----------------

### References

[1] *[Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation](http://www.mit.edu/~adalca/files/papers/cvpr2018_priors.pdf)* \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
CVPR 2018

[2] *[Unsupervised Data Imputation via Variational Inference of Deep Subspaces](https://arxiv.org/abs/1903.03503)* \
Adrian V. Dalca, John Guttag, Mert R. Sabuncu \
Arxiv preprint 2019
