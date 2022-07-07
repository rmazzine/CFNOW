"""
This script has the strategies to segment images
"""
from skimage.segmentation import quickshift


def gen_quickshift(img, params_seg):
    """
    Simple wrapper of the Scikit-Image quickshift algorithm
    :param img: Image to be segmented
    :type img: np.ndarray
    :param params_seg: Parameters to be passed to the segmentation algorithm
    :type params_seg: dict
    :return: An array with a segment number for each pixel
    :rtype: np.ndarray
    """
    if params_seg is None:
        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}

    return quickshift(img, **params_seg)
