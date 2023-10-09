from typing import Tuple, List
from pathlib import Path

import pytest
import tensorflow as tf
import numpy as np

from SynthSeg.brain_generator import BrainGenerator, read_tfrecords
from . import TestData
from SynthSeg.model_inputs import build_model_inputs
from dataclasses import dataclass


@dataclass
class TestTFRecord:
    path: Path
    shape: Tuple[int, int, int, int]
    n_labels: int
    label_maps: List[str]
    target_res: int
    dataset: tf.data.Dataset


@pytest.fixture(scope="session")
def model_inputs():
    """This fixture returns a list of inputs necessary for the label_to_image model.
    Returns:
        A List containing the input label map, as well as the means and stds defining the parameters of the GMM.
    """
    return get_model_inputs()


def get_model_inputs():
    """
    Create label_to_image input.
    Use this method if you need to call it directly in an interactive session.
    Returns:
        label_to_image input
    """
    labels_classes_path = TestData.synth_seg_path / "data" / "labels_classes_priors"
    generation_labels = np.load(labels_classes_path / "generation_labels.npy")
    generation_classes = np.load(labels_classes_path / "generation_classes.npy")
    model_inputs_generator = build_model_inputs(
        path_label_maps=sorted(TestData.get_label_maps()),
        n_labels=len(generation_labels),
        batchsize=1,
        n_channels=1,
        subjects_prob=None,
        generation_classes=generation_classes,
        prior_means=None,
        prior_stds=None,
        prior_distributions="uniform",
        use_specific_stats_for_channel=False,
        mix_prior_and_random=False,
    )
    return next(model_inputs_generator)


@pytest.fixture(scope="session")
def tfrecord(tmp_path_factory) -> TestTFRecord:
    """
    Returns:
        A TestTFRecord instance that contains information about its generation process.
    """
    tf.keras.utils.set_random_seed(43)

    label_map_files = TestData.get_label_maps()

    brain_generator = BrainGenerator(label_map_files[0], target_res=8, batchsize=2)
    tfrecord_path = tmp_path_factory.mktemp("tfrecords") / "test.tfrecord"
    brain_generator.generate_tfrecord(tfrecord_path)
    dataset = read_tfrecords([tfrecord_path])

    return TestTFRecord(tfrecord_path, (32, 32, 32, 1), 53, label_map_files, 8, dataset)
