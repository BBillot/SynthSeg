from dataclasses import dataclass
import numpy as np
import os
import re


@dataclass
class FreeSurferLUTEntry:
    """
    Singe entry from the FreeSurfer color map
    """
    name: str = "Unknown"
    label: int = 0
    rgba: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])


@dataclass
class TissueType:
    """
    Configuration for the synthesis of a single tissue type.
    It contains all information about how one tissue type is
    synthesized in the fake MRI images.
    """
    label: FreeSurferLUTEntry = FreeSurferLUTEntry()
    segmentation_class: int = 0
    mean: float = 0.0
    std_dev: float = 0.0


def getFreeSurferLUT() -> dict:
    """
    Provides all default labels of FreeSurfer as a dictionary.

    Returns:
        dict: Mapping of label id to `FreeSurferLUTEntry`
    """
    lut_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "freeSurferLUT.txt")
    assert os.path.isfile(lut_file), f"Tried to find file {lut_file}"
    result = {}
    with open(lut_file) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0 or stripped[0] == "#":
            continue

        parts = stripped.split()
        assert len(parts) == 6

        labelId = int(parts[0])
        name = parts[1]
        rgba = np.array([
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5])
        ])

        assert labelId not in result
        result[labelId] = FreeSurferLUTEntry(name, labelId, rgba)
    return result


FSL_LUT = getFreeSurferLUT()
FSL_LEFT_LABEL_REGEX = re.compile(r"^([Ll])eft[_-]|^ctx-(lh)-|^wm[_-](lh)[_-]|(_l)$")
FSL_RIGHT_LABEL_REGEX = re.compile(r"^([Rr])ight[_-]|^ctx-(rh)-|^wm[_-](rh)[_-]|(_r)$")


def substitute_left_right(match) -> str:
    if match.group(1):
        if match.group(1) == "L":
            return match.group().replace("Left", "Right")
        else:
            return match.group().replace("left", "right")
    elif match.group(2) or match.group(3):
        return match.group().replace("lh", "rh")
    elif match.group(4):
        return match.group().replace("_l", "_r")
    else:
        return match.group()


def substitute_right_left(match) -> str:
    if match.group(1):
        if match.group(1) == "R":
            return match.group().replace("Right", "Left")
        else:
            return match.group().replace("right", "left")
    elif match.group(2) or match.group(3):
        return match.group().replace("rh", "lh").replace("rh", "lh")
    elif match.group(4):
        return match.group().replace("_r", "_l")
    else:
        return match.group()


def generate_tissue_types_from_sample(scan_data: np.ndarray, segmentation_data: np.ndarray, label: int) -> TissueType:
    """
    Takes an existing segmentation for a scan and calculates statistical values for the segmentation class of the given
    `label`.
    The segmentation is expected to have class-labels according to FreeSurfer's brain segmentations.

    Args:
        scan_data (np.ndarray): The original brain-scan or an MPM map like PD, T1, etc.
        segmentation_data (np.ndarray): The segmentation of `scan_data`. Must have the same shape as `scan_data`.
        label: The tissue class to calculate the statistics for.

    Returns:
        TissueType: Statistics of the region with additional FreeSurfer metadata.
    """
    scan_data = scan_data
    seg_data = segmentation_data
    mask = seg_data == label
    data = scan_data[mask]
    mean = np.mean(data)
    std_dev = np.std(data)
    if label not in FSL_LUT.keys():
        print(f"Label number {label} not found in FSL lookup table. Using background for it!")
        label = 0
    lut_entry = FSL_LUT[label]
    return TissueType(lut_entry, label, mean, std_dev)
