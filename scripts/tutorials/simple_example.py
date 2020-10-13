# Very simple script showing how to generate new images with lab2im

import os
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator


# path of the input label map
path_label_map = '../../data/training_label_maps/subject01_seg.nii.gz'
# path where to save the generated image
result_dir = '../../generated_images'

# generate an image from the label map.
# Because the image is spatially deformed, we also output the corresponding deformed label map.
brain_generator = BrainGenerator(path_label_map)
im, lab = brain_generator.generate_brain()

# save output image and label map
utils.mkdir(result_dir)
utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                  os.path.join(result_dir, 'brain.nii.gz'))
utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                  os.path.join(result_dir, 'labels.nii.gz'))
