from dataclasses import dataclass
import numpy as np
import os


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


def generateTissueTypesFromSample(scan_data: np.ndarray, segmentation_data: np.ndarray, label: int) -> TissueType:
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
    scanData = scan_data
    segData = segmentation_data
    mask = segData == label
    data = scanData[mask]
    mean = np.mean(data)
    stdDev = np.std(data)
    lut_entry = FSL_LUT[label] if label in FSL_LUT.keys() else FreeSurferLUTEntry()
    return TissueType(lut_entry, label, mean, stdDev)
