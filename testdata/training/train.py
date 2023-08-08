from simple_parsing import parse
from SynthSeg.training_with_tfrecords import TrainingOptions, training


if __name__ == "__main__":
    opts = parse(TrainingOptions)
    # opts = TrainingOptions(
    #     data_dir="./",
    #     output_dir="./output",
    #     wl2_epochs=1,
    #     dice_epochs=1,
    #     steps_per_epoch=5,
    # )
    training(opts)
