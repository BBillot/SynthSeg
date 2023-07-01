from dataclasses import dataclass

import simple_parsing
import os
import glob

from SynthSeg.training import training_from_options
from SynthSeg.training_options import TrainingOptions


@dataclass
class CmdLineTrainingOptions:
    config_file: str = ""
    """
    Path to a JSON file that represents the serialized form of a TrainingOptions instance.
    """


if __name__ == "__main__":
    args: CmdLineTrainingOptions = simple_parsing.parse(CmdLineTrainingOptions)
    file_name = args.config_file

    # Check if the configuration file exists and load it
    if (not os.path.isfile(file_name)) or (not file_name.endswith(".json")) or (not os.access(file_name, os.R_OK)):
        raise RuntimeError(f"Configuration file {file_name} does not exist or is not readable.")
    training_options = TrainingOptions.load(file_name)

    # If we have a checkpoint model, use that for continuing the training
    model_dir = training_options.model_dir
    checkpoint_files = glob.glob(f"{model_dir}/*.h5")
    if len(checkpoint_files) > 0:
        # Todo: No clue if _all_ checkpoints are save or only the last one!
        training_options.checkpoint = checkpoint_files[0]
        training_options.dice_epochs = 0

    print(training_options)
    training_from_options(**training_options.to_dict())
