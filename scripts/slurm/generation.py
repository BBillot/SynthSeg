import os.path
from dataclasses import dataclass
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
    """Output Folder where the generated maps are stored."""

    config_file: Optional[str] = None
    """Path to the JSON config file containing the GeneratorOptions for the brain generator."""

    count: int = 1
    """Number of training pairs that are created."""


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


if __name__ == '__main__':
    logger = logging.getLogger("Generate Brain")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
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
        utils.mkdir(output_dir)
        utils.mkdir(f"{output_dir}/images")
        utils.mkdir(f"{output_dir}/labels")
    else:
        logger.error(f"No output directory specified")
        exit(-1)

    conf_file = fix_relative_path(general_params.config_file)
    if isinstance(conf_file, str) and isfile(conf_file) and access(conf_file, R_OK):
        logger.info("Loading generator config from configuration file.")

        # Load config and make paths within the config absolute
        generator_config = GeneratorOptions.load(conf_file)
        generator_config = generator_config.with_absolute_paths(os.path.abspath(conf_file))
    else:
        logger.error("No valid config file. Initialize generator with default values and cmd-line parameters.")
        generator_config = args.generate
        generator_config = generator_config.with_absolute_paths(os.path.abspath(conf_file))

    if general_params.count <= 0:
        logger.error(f"Number of training pairs to generate must be positive but was {general_params.count}.")
        exit(0)

    from SynthSeg.brain_generator import create_brain_generator
    generator = create_brain_generator(generator_config)
    image_output_dir = f"{output_dir}/images"
    label_output_dir = f"{output_dir}/labels"

    for i in range(general_params.count):
        image, label = generator.generate_brain()
        utils.save_volume(image, generator.aff, generator.header, f"{image_output_dir}/{str(i).zfill(6)}.nii.gz")
        utils.save_volume(label, generator.aff, generator.header, f"{label_output_dir}/{str(i).zfill(6)}.nii.gz")
        logger.info(f"Exported training pair number {str(i).zfill(6)}")
