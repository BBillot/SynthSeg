"""
Very simple script to generate an example of the synthetic data used to train SynthSeg.
This is for visualisation purposes, since it uses all the default parameters.
"""

from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# generate an image from the label map.
brain_generator = BrainGenerator('../../data/training_label_maps/training_seg_01.nii.gz')
im, lab = brain_generator.generate_brain()

# save output image and label map
utils.save_volume(im, brain_generator.aff, brain_generator.header, './generated_examples/image_default.nii.gz')
utils.save_volume(lab, brain_generator.aff, brain_generator.header, './generated_examples/labels_default.nii.gz')
