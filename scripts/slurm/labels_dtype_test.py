from simple_parsing import parse
from dataclasses import dataclass
from SynthSeg.brain_generator import create_brain_generator
from SynthSeg.brain_generator_options import GeneratorOptions
from SynthSeg.training_options import TrainingOptions
from SynthSeg.training_with_tfrecords import training
from pathlib import Path
import tensorflow as tf


@dataclass
class Args:
    mode: str = "generate"
    """
    One of "generate" or "train"
    """

    config: str = "./config.yml"
    """
    Path to the corresponding config file.
    """

    output: str = "./tfrecords"
    """
    Output path for the tfrecord files.
    """

    dtype: str = "int32"
    """
    DType of the labels
    """


def generate(args: Args):
    opts = GeneratorOptions.load_yaml(args.config)
    brain_generator = create_brain_generator(opts)

    brain_generator.generate_tfrecord(
        Path(args.output) / "test.tfrecord", labels_dtype=str_to_dtype(args.dtype)
    )


def train(args: Args):
    opts = TrainingOptions.load_yaml(args.config)
    opts.tfrecords_dir = args.output

    results = training(opts, str_to_dtype(args.dtype))
    print(results.history)


def str_to_dtype(dtype: str):
    if dtype == "int32":
        return tf.int32
    if dtype == "uint8":
        return tf.uint8

    raise ValueError(f"dtype '{dtype}' not implemented!")


if __name__ == "__main__":
    args: Args = parse(Args)

    tf.keras.utils.set_random_seed(43)
    if args.mode == "generate":
        generate(args)
    elif args.mode == "train":
        train(args)
    else:
        raise ValueError(f"Mode '{args.mode}' not supported!")