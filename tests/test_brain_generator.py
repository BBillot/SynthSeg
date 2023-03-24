import tensorflow as tf
from . import TestData
import SynthSeg.brain_generator as bg


def test_brain_generator():
    tf.config.run_functions_eagerly(True)
    label_map_files = TestData.get_label_maps()
    brain_generator = bg.BrainGenerator(label_map_files[0])
    im, lab = brain_generator.generate_brain()
    assert im.shape == lab.shape, "Shape of the label image and the generated MRI are not the same"



