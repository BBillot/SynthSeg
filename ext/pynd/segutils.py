import numpy as np
import scipy.ndimage


def seg2contour(seg, exclude_zero=True, contour_type='inner', thickness=1):
    """
    transform nd segmentation (label maps) to contour maps

    Parameters
    ----------
    seg : nd array
        volume of labels/segmentations
    exclude_zero : optional logical
        whether to exclude the zero label.
        default True
    contour_type : string
        where to draw contour voxels relative to label 'inner','outer', or 'both'
    thickness
    """

    # extract unique labels
    labels = np.unique(seg)
    if exclude_zero:
        labels = np.delete(labels, np.where(labels == 0))

    # get the contour of each label
    contour_map = seg * 0
    for lab in labels:

        # extract binary label map for this label
        label_map = seg == lab

        # extract contour map for this label
        thickness = thickness + 0.01
        label_contour_map = bw2contour(label_map, contour_type=contour_type, thr=thickness)

        # assign contour to this label
        contour_map[label_contour_map] = lab

    return contour_map


def bw2contour(bwvol, contour_type='both', thr=1.01):
    """computes the contour of island(s) on a nd logical volume"""

    # obtain a signed distance transform for the bw volume
    sdtrf = bw2sdtrf(bwvol)

    if contour_type == 'inner':
        return np.logical_and(sdtrf <= 0, sdtrf > -thr)
    elif contour_type == 'outer':
        return np.logical_and(sdtrf >= 0, sdtrf < thr)
    else:
        assert contour_type == 'both', 'type should only be inner, outer or both'
        return np.abs(sdtrf) < thr


def bw2sdtrf(bwvol):
    """computes the signed distance transform from the surface between the binary elements of logical bwvol"""

    # get the positive transform (outside the positive island)
    posdst = bwdist(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = bwdist(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def bwdist(bwvol):
    """positive distance transform from positive entries in logical image"""

    # reverse volume to run scipy function
    revbwvol = np.logical_not(bwvol)

    # get distance
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)
