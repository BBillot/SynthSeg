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


def test_left_right_names():
    names = [entry.name for entry in list(fsl_tools.FSL_LUT.values())]
    for entry in names:
        left_name: str = entry
        if fsl_tools.FSL_LEFT_LABEL_REGEX.match(left_name):
            right_name_1 = left_name.replace("Left", "Right")
            right_name_2 = left_name.replace("ctx_lh", "ctx_rh")
            right_name_3 = left_name.replace("ctx-lh", "ctx-rh")
            right_name_4 = left_name.replace("wm-lh", "wm-rh")
            right_name_5 = left_name.replace("wm_lh", "wm_rh")
            right_name_6 = left_name.replace("_l", "_r")
            assert right_name_1 in names or \
                   right_name_2 in names or \
                   right_name_3 in names or \
                   right_name_4 in names or \
                   right_name_5 in names or \
                   right_name_6 in names, f"Missing right equivalent region for {left_name}"


def test_right_left_names():
    names = [entry.name for entry in list(fsl_tools.FSL_LUT.values())]
    for entry in names:
        right_name: str = entry
        if fsl_tools.FSL_RIGHT_LABEL_REGEX.match(right_name):
            left_name_1 = right_name.replace("Right", "Left")
            left_name_2 = right_name.replace("ctx_rh", "ctx_lh")
            left_name_3 = right_name.replace("ctx-rh", "ctx-lh")
            left_name_4 = right_name.replace("wm-rh", "wm-lh")
            left_name_5 = right_name.replace("wm_rh", "wm_lh")
            left_name_6 = right_name.replace("_r", "_l")
            assert left_name_1 in names or \
                   left_name_2 in names or \
                   left_name_3 in names or \
                   left_name_4 in names or \
                   left_name_5 in names or \
                   left_name_6 in names, f"Missing left equivalent region for {right_name}"
