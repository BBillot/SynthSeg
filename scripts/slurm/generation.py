import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from simple_parsing import ArgumentParser
from os import access, R_OK, path
from os.path import isfile
import logging
import sys

from SynthSeg.brain_generator_options import GeneratorOptions

project_directory = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


@dataclass
class Options:
    output_dir: Optional[str] = None
    """Output Folder where the generated training data is stored."""

    config_file: Optional[str] = None
    """Path to the JSON config file containing the GeneratorOptions for the brain generator."""

    count: int = 1
    """Number of training files that are created."""

    start_int: int = 0
    """The file names of the generated training data are simply numbers (6 digits). Start at this int."""

    tfrecord: bool = False
    """Store the generated image-label pairs as TFRecords."""


def fix_relative_path(dir_or_file: str) -> str:
    """
    Turns a relative path into an absolute one that has its root at the project directory
    """
    if len(dir_or_file) == 0:
        return "/"
    elif len(dir_or_file) > 0 and dir_or_file[0] != "/":
        return f"{project_directory}/{dir_or_file}"
    else:
        return dir_or_file


def _generate_image_label_pair(
    brain_generator: "SynthSeg.brain_generator.BrainGenerator",
    output_path: Path,
    file_name: str,
):
    """Helper function to generate and save Image-Label pairs in the `.nii.gz` format for training.

    Args:
        brain_generator: An instance of the `SynthSeg.brain_generator.BrainGenerator`.
        output_path: Path to the output directory. We will create the subdirectories `images` and `labels`.
        file_name: Name of the output files. We will suffix them with `.nii.gz`.
    """
    image_output_path = output_path / "images"
    image_output_path.mkdir(parents=True, exist_ok=True)

    labels_output_path = output_path / "labels"
    labels_output_path.mkdir(parents=True, exist_ok=True)

    image, label = brain_generator.generate_brain()
    utils.save_volume(
        image,
        brain_generator.aff,
        brain_generator.header,
        str(image_output_path / f"{file_name}.nii.gz"),
    )
    utils.save_volume(
        label,
        brain_generator.aff,
        brain_generator.header,
        str(labels_output_path / f"{file_name}.nii.gz"),
    )


def _generate_tfrecord(
    brain_generator: "SynthSeg.brain_generator.BrainGenerator",
    output_path: Path,
    file_name: str,
):
    """Helper class to generate and save a TFRecord for training.

    Args:
        brain_generator: An instance of the `SynthSeg.brain_generator.BrainGenerator`.
        output_path: Path to the output directory. We will create the subdirectory `tfrecords`.
        file_name: Name of the output file. We will suffix it with `.tfrecord`.
    """
    output_path = output_path / "tfrecords"
    output_path.mkdir(parents=True, exist_ok=True)

    brain_generator.generate_tfrecord(output_path / f"{file_name}.tfrecord")


if __name__ == "__main__":
    logger = logging.getLogger("Generate Brain")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(Options, "general")
    # noinspection PyTypeChecker
    parser.add_arguments(GeneratorOptions, "generate")
    args = parser.parse_args()

    from ext.lab2im import utils

    general_params: Options = args.general
    if isinstance(general_params.output_dir, str):
        output_dir = fix_relative_path(general_params.output_dir)
    else:
        logger.error(f"No output directory specified")
        exit(-1)

    conf_file = fix_relative_path(general_params.config_file)
    if isinstance(conf_file, str) and isfile(conf_file) and access(conf_file, R_OK):
        logger.info("Loading generator config from configuration file.")

        # Load config and make paths within the config absolute
        generator_config = GeneratorOptions.load(conf_file)
        generator_config = generator_config.with_absolute_paths(
            os.path.abspath(conf_file)
        )
    else:
        logger.error(
            "No valid config file. Initialize generator with default values and cmd-line parameters."
        )
        generator_config = args.generate
        generator_config = generator_config.with_absolute_paths(
            os.path.abspath(conf_file)
        )

    if general_params.count <= 0:
        logger.error(
            f"Number of training pairs to generate must be positive but was {general_params.count}."
        )
        exit(0)

    from SynthSeg.brain_generator import create_brain_generator

    generator = create_brain_generator(generator_config)
    for i in range(
        general_params.start_int, general_params.start_int + general_params.count
    ):
        file_number = str(i).zfill(6)

        if general_params.tfrecord:
            _generate_tfrecord(generator, Path(output_dir), file_number)
        else:
            _generate_image_label_pair(generator, Path(output_dir), file_number)

        logger.info(f"Exported training pair number {file_number}")
