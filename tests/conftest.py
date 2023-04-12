import pytest
import tensorflow as tf
import numpy as np
import os
import random


def set_seeds():
    seed = 12345
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism():
    set_seeds()

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


@pytest.fixture
def fixed_random_seed():
    set_global_determinism()
