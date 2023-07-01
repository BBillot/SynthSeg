import pathlib
import os.path
from typing import List
import subprocess


def get_git_branch() -> str:
    """
    Tries to extract the current git branch of the project.
    We'll use that to differentiate output paths of test runs so that we can compare results across branches.
    Returns:

    """
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        branch = "Unknown"
    return branch


class TestData:
    """
    Helper class for everything related to input and output of test data.

    Attributes:
        debug_nifti_output Set to true if test methods are allowed to write out debug nifti files
        synth_seg_path Path of the project
        git_branch Branch name of the current git branch
    """
    debug_nifti_output = True
    synth_seg_path = pathlib.Path(__file__).parent.parent.resolve()
    git_branch = get_git_branch()

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
    def get_test_data_dir(cls) -> pathlib.Path:
        """
        Returns the directory path for test data.

        Returns:
            pathlib.Path: The absolute path of the test data directory.
        """
        return (TestData.synth_seg_path / "testdata").absolute()

    @classmethod
    def get_test_output_dir(cls) -> pathlib.Path:
        """
        Provides (and ensures it exists) a temporary test output directory inside the project
        folder and creates it otherwise.

        Returns:
            The absolute path to the temporary test output folder in the project directory
        """
        output_dir = (TestData.synth_seg_path / "test_output" / TestData.git_branch).absolute()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
