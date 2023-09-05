import SynthSeg.analysis.freesurfer_tools as fsl_tools


def test_left_right_regions():
    names = [entry.name for entry in list(fsl_tools.FSL_LUT.values())]
    left_regions = []
    right_regions = []
    neutral_regions = []
    for entry in names:
        if fsl_tools.FSL_LEFT_LABEL_REGEX.match(entry):
            left_regions.append(entry)
        elif fsl_tools.FSL_RIGHT_LABEL_REGEX.match(entry):
            right_regions.append(entry)
        else:
            neutral_regions.append(entry)
    assert len(left_regions) == len(right_regions), \
        (f"Left/Right regions don't have the same number of elements:"
         f"{len(left_regions)}/{len(right_regions)}")


def test_right_left_names():
    all_names = [entry.name for entry in fsl_tools.FSL_LUT.values()]
    for name in all_names:
        if fsl_tools.FSL_LEFT_LABEL_REGEX.match(name):
            right_name = fsl_tools.FSL_LEFT_LABEL_REGEX.sub(fsl_tools.substitute_left_right, name)
            assert right_name in all_names, \
                f"For {name}, right counterpart region '{right_name}' doesn't exist in FSL LUT."
        elif fsl_tools.FSL_RIGHT_LABEL_REGEX.match(name):
            left_name = fsl_tools.FSL_RIGHT_LABEL_REGEX.sub(fsl_tools.substitute_right_left, name)
            assert left_name in all_names, \
                f"For {name}, left counterpart region '{left_name}' doesn't exist in FSL LUT."


def test_label_name_replacement():
    left_samples = [
        "Left-Cerebral-Exterior",
        "IPL_PFcm_l",
        "left_hippocampal_fissure",
        "ctx-lh-parsopercularis",
        "wm-lh-superiorfrontal",
        "ctx-lh-entorhinal"
    ]

    right_samples = [
        "Right-Cerebral-Exterior",
        "IPL_PFcm_r",
        "right_hippocampal_fissure",
        "ctx-rh-parsopercularis",
        "wm-rh-superiorfrontal",
        "ctx-rh-entorhinal"
    ]

    for [left, right] in zip(left_samples, right_samples):
        substituted_left = fsl_tools.FSL_LEFT_LABEL_REGEX.sub(fsl_tools.substitute_left_right, left)
        substituted_right = fsl_tools.FSL_RIGHT_LABEL_REGEX.sub(fsl_tools.substitute_right_left, right)
        assert right == substituted_left
        assert left == substituted_right
