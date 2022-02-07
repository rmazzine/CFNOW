from skimage.segmentation import quickshift


def gen_quickshift(img, params_seg):
    if params_seg is None:
        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}

    return quickshift(img)
