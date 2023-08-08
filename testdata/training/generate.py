from SynthSeg.brain_generator import create_brain_generator
from SynthSeg.brain_generator_options import GeneratorOptions
from simple_parsing import parse


if __name__ == "__main__":
    opts: GeneratorOptions = parse(GeneratorOptions)
    # opts = GeneratorOptions()
    brain_generator = create_brain_generator(opts)

    for i in range(1):
        file_name = f"./64_{i}"
        brain_generator.generate_tfrecord(file_name, n=5)
