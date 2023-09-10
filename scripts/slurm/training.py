import simple_parsing
import os
from dataclasses import dataclass

from SynthSeg.training_options import TrainingOptions


@dataclass
class CmdLineTrainingOptions:
    """
    Specify the training configuration
    """

    cfg_file: str = ""
    """
    Path to a JSON file that represents the serialized form of a TrainingOptions instance.
    """


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("--cfg_file", type=str, default="", help="Path to a cfg file.")
    parser.add_arguments(TrainingOptions, dest="options")
    args = parser.parse_args()

    if args.cfg_file:
        training_options = TrainingOptions.load(args.cfg_file)
        training_options = training_options.with_absolute_paths(os.path.abspath(args.cfg_file))
    else:
        training_options = args.options

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
