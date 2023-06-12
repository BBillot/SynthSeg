from dataclasses import dataclass
from typing import Optional
from simple_parsing import ArgumentParser
from os import access, R_OK
from os.path import isfile

from SynthSeg.brain_generator_options import GeneratorOptions
from SynthSeg.brain_generator import BrainGenerator
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
    parser = ArgumentParser()
    parser.add_arguments(Options, "general")
    parser.add_arguments(GeneratorOptions, "generate")
    args = parser.parse_args()

    general_params: Options = args.general
    if isinstance(general_params.output_dir, str):
        utils.mkdir(general_params.output_dir)

    conf_file = general_params.config_file
    if isinstance(conf_file, str) and isfile(conf_file) and access(conf_file, R_OK):
        print("Loading generator config from configuration file.")
        generator_config = GeneratorOptions.load(conf_file)
    else:
        print("No valid config file. Initialize generator with default values and cmd-line parameters.")
        generator_config = args.generate

    print(generator_config)

    # TODO: Create brain generator from this config. Maybe we can merge cmg args into a loaded config
