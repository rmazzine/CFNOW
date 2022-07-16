import unittest
from unittest.mock import patch

import numpy as np

from cfnow._img_segmentation import gen_quickshift


class TestScriptBase(unittest.TestCase):

    # Simple image to be used in tests
    img = []
    for ir in range(240):
        row_img = []
        row_segments = []
        row_replace_image = []
        for ic in range(240):
            row_replace_image.append([0, 255, 0])
            if ir > ic:
                row_img.append([255, 0, 0])
                row_segments.append(1)
            else:
                row_img.append([255, 255, 255])
                row_segments.append(0)
        img.append(row_img)
    img = np.array(img).astype('uint8')

    @patch('cfnow._img_segmentation.quickshift')
    def test_gen_quickshift_params_none(self, mock_quickshift):
        # Assert if it was called with default (kernel_size=4, max_dist=200, ratio=0.2) params if params_seg is None

        gen_quickshift(self.img, None)

        mock_quickshift.assert_called_once_with(self.img, kernel_size=4, max_dist=200, ratio=0.2)

    @patch('cfnow._img_segmentation.quickshift')
    def test_gen_quickshift_params_defined(self, mock_quickshift):
        # Assert if it was called with defined parameters in params_seg

        params_seg = {'kernel_size': 10, 'max_dist': 1000, 'ratio': 10}

        gen_quickshift(self.img, params_seg)

        mock_quickshift.assert_called_once_with(self.img, kernel_size=10, max_dist=1000, ratio=10)
