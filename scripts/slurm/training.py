import simple_parsing
import os
from dataclasses import dataclass

from SynthSeg.training_options import TrainingOptions


@dataclass
class CmdLineTrainingOptions:
    """
    Specify the training configuration
    """

    training_config: str = ""
    """
    Path to a JSON file that represents the serialized form of a TrainingOptions instance.
    """


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_arguments(CmdLineTrainingOptions, dest="config")
    args: CmdLineTrainingOptions = parser.parse_args().config
    file_name = args.config_file

    if file_name == "":
        print("Missing config file")
        parser.print_help()
        exit(1)

    # Check if the configuration file exists and load it
    if (
        (not os.path.isfile(file_name))
        or (not file_name.endswith(".json"))
        or (not os.access(file_name, os.R_OK))
    ):
        raise RuntimeError(
            f"Configuration file {file_name} does not exist or is not readable."
        )

    # Loading the training options and fixing all relative paths
    training_options = TrainingOptions.load(file_name)
    training_options = training_options.with_absolute_paths(os.path.abspath(file_name))

    # TODO: Fix this
    # If we have a checkpoint model, use that for continuing the training
    # model_dir = training_options.model_dir
    # checkpoint_files = glob.glob(f"{model_dir}/*.h5")
    # if len(checkpoint_files) > 0:
    #     training_options.checkpoint = checkpoint_files[0]
    #     training_options.dice_epochs = 0

    # We import it here so that TF will be loaded after everything has been set up.
    # This should help to not wait 20 seconds when you just call it with --help.
    if training_options.tfrecords_dir:
        from SynthSeg.training_with_tfrecords import training

        training(training_options)
    else:
        from SynthSeg.training import training_from_options

        training_from_options(training_options)
