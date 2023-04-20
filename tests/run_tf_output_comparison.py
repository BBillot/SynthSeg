import os
import glob
from typing import List
from . import TestData


def find_branch_folders() -> List[str]:
    test_directory = TestData.get_test_output_dir().parent
    subs = next(os.walk(test_directory))[1]
    assert len(subs) == 2, "The test output folder must contain exactly 2 subdirectories for comparison"
    return list(map(lambda d: (test_directory / d).as_posix(), subs))


def create_comparison_input(folders: list[str]) -> List[dict]:
    assert len(folders) == 2
    f1 = folders[0]
    f2 = folders[1]
    nii_files = glob.glob(os.path.join(f1, "*.nii"))
    npy_files = glob.glob(os.path.join(f1, "*.npy"))
    result = []
    for file_type in [["nii", nii_files], ["npy", npy_files]]:
        current_id = file_type[0]
        current_files = file_type[1]
        for f in current_files:
            file_name = os.path.basename(f)
            other_file = os.path.join(f2, file_name)
            if os.path.exists(other_file):
                result.append({"type": current_id, "f1": f, "f2": other_file})
            else:
                assert False, f"The file {f} does not exist in directory {f2}"

    return result

