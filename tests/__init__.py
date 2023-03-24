import pathlib
import os.path
from typing import List


class TestData:
    """
    Helper class for everything related to input and output of test data.

    Attributes:
        debug_nifti_output  Set to true if test methods are allowed to write out debug nifti files
    """
    debug_nifti_output = True

    synth_seg_path = pathlib.Path(__file__).parent.parent.resolve()

    @classmethod
    def get_label_maps(cls) -> List[str]:
        """
        Provides the absolute paths of the training label maps.
        Returns:
            List of all training label map paths
        """
        label_dir = TestData.synth_seg_path / "data" / "training_label_maps"
        return list(map(lambda p: p.absolute().as_posix(), label_dir.glob("*.nii.gz")))

    @classmethod
    def get_tmp_output_dir(cls) -> pathlib.Path:
        """
        Provides (and ensures it exist) a temporary test output directory inside the project
        folder.
        :return: (Creates and) returns the absolute path to "tmp_test_output" folder in the project directory
        """
        output_dir = (TestData.synth_seg_path / "tmp_test_output").absolute()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir
