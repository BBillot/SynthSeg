# python imports
import os
import logging
import numpy as np
from scipy.ndimage.interpolation import affine_transform
# project imports
from SynthSeg.vae_model import build_decoder
# third-party imports
from ext.lab2im import utils, edit_volumes
# disable tensorflow warnings
logging.getLogger('tensorflow').disabled = True

# inputs
n_examples = 50
vae_dir = '../../VAE'
result_dir = '../generated_images'
image_shape = [224, 224, 224]
latent_shape = [1, 5, 7, 5, 16]

# ----------------------------------------------------------------------------------------------------------------------

# create result directory
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

# load template
im, im_shape, aff, _, _, h, _ = utils.get_volume_info(os.path.join(vae_dir, 'template.nii'), return_volume=True)

# build keras model
decoder = build_decoder(latent_shape[1:], image_shape)
decoder.load_weights(os.path.join(vae_dir, 'decoder_keras_model.h5'))

# Get some info about the VAE
VAEInfo = np.load(os.path.join(vae_dir, 'VAEInfo.npz'))
subjectToTrainMat = np.linalg.inv(VAEInfo['trainToTemplateMat'])

for idx in range(n_examples):
    utils.print_loop_info(idx, n_examples, spacing=1)

    # sample lesion
    code = np.float32(np.random.normal(size=latent_shape))
    lesion = np.squeeze(decoder.predict(code))

    # bring sampled lesion to subject space
    lesion = affine_transform(lesion, subjectToTrainMat, output_shape=image_shape, order=1)
    lesion = edit_volumes.crop_volume(lesion, cropping_shape=im_shape)

    # build lesion mask and save it
    threshold = np.power(10, -np.random.uniform(-1, 5))
    lesion = (lesion > threshold) * 77
    utils.save_volume(lesion, aff, h, os.path.join(result_dir, str(idx + 1) + '.nii.gz'))
