from SynthSeg.training_options import TrainingOptions
from SynthSeg.brain_generator_options import GeneratorOptions

from . import TestData


def test_float_str_bool_deserialization():
    """
    This specifically tests if the bool/float/str types are deserialized correctly
    """
    data_dir = TestData.get_test_data_dir()
    config_1 = str(data_dir / "serialization" / "training_scal_bounds_false.json")
    opts = TrainingOptions.load(config_1)
    assert opts.scaling_bounds is False
    config_2 = str(data_dir / "serialization" / "training_scal_bounds_0.json")
    opts = TrainingOptions.load(config_2)
    assert opts.scaling_bounds == 0.0
    config_3 = str(data_dir / "serialization" / "training_scal_bounds_str.json")
    opts = TrainingOptions.load(config_3)
    assert opts.scaling_bounds == "lorem"


def test_none_deserialization():
    data_dir = TestData.get_test_data_dir()
    config_1 = str(data_dir / "serialization" / "training_gen_labels_none.json")
    opts = TrainingOptions.load(config_1)
    assert opts.generation_labels is None
    config_2 = str(data_dir / "serialization" / "training_gen_labels_str.json")
    opts = TrainingOptions.load(config_2)
    assert opts.generation_labels == "hello"
    config_3 = str(data_dir / "serialization" / "training_gen_labels_list_int.json")
    opts = TrainingOptions.load(config_3)
    assert opts.generation_labels == [1, 2, 3, 4]


def test_none_str_int_list():
    data_dir = TestData.get_test_data_dir()
    config_1 = str(data_dir / "serialization" / "training_output_shape_none.json")
    opts = TrainingOptions.load(config_1)
    assert opts.output_shape is None
    config_2 = str(data_dir / "serialization" / "training_output_shape_str.json")
    opts = TrainingOptions.load(config_2)
    assert opts.output_shape == "path/to/nparray"
    config_3 = str(data_dir / "serialization" / "training_output_shape_list.json")
    opts = TrainingOptions.load(config_3)
    assert opts.output_shape == [160, 160, 160]
    config_4 = str(data_dir / "serialization" / "training_output_shape_int.json")
    opts = TrainingOptions.load(config_4)
    assert opts.output_shape == 160


def test_generator_options_deserialization():
    data_dir = TestData.get_test_data_dir()
    config_1 = str(data_dir / "serialization" / "generator.json")
    opts = GeneratorOptions.load(config_1)
    assert opts.thickness is None
