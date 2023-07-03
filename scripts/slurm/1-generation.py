from dataclasses import dataclass
from typing import Optional
from simple_parsing import ArgumentParser
from os import access, R_OK
from os.path import isfile
import logging
import sys

from SynthSeg.brain_generator_options import GeneratorOptions
from SynthSeg.brain_generator import create_brain_generator
from ext.lab2im import utils


@dataclass
class Options:
    output_dir: Optional[str] = None
    """Output Folder where the generated maps are stored."""

    config_file: Optional[str] = None
    """Path to the JSON config file containing the GeneratorOptions for the brain generator."""

    count: int = 1
    """Number of training pairs that are created."""


if __name__ == '__main__':
    logger = logging.getLogger("Generate Brain")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = ArgumentParser()
    parser.add_arguments(Options, "general")
    parser.add_arguments(GeneratorOptions, "generate")
    args = parser.parse_args()

    general_params: Options = args.general
    if isinstance(general_params.output_dir, str):
        utils.mkdir(general_params.output_dir)
        utils.mkdir(f"{general_params.output_dir}/images")
        utils.mkdir(f"{general_params.output_dir}/labels")
    else:
        logger.error(f"No output directory specified")
        exit(-1)

    conf_file = general_params.config_file
    if isinstance(conf_file, str) and isfile(conf_file) and access(conf_file, R_OK):
        logger.info("Loading generator config from configuration file.")
        generator_config = GeneratorOptions.load(conf_file)
    else:
        logger.info("No valid config file. Initialize generator with default values and cmd-line parameters.")
        generator_config = args.generate

    if general_params.count <= 0:
        logger.error(f"Number of training pairs to generate must be positive but was {general_params.count}.")
        exit(0)

    generator = create_brain_generator(generator_config)
    image_output_dir = f"{general_params.output_dir}/images"
    label_output_dir = f"{general_params.output_dir}/labels"

    for i in range(general_params.count):
        image, label = generator.generate_brain()
        utils.save_volume(image, generator.aff, generator.header, f"{image_output_dir}/{str(i).zfill(6)}.nii.gz")
        utils.save_volume(label, generator.aff, generator.header, f"{label_output_dir}/{str(i).zfill(6)}.nii.gz")
        logger.info(f"Exported training pair number {str(i).zfill(6)}")
