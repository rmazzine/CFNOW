import unittest

import numpy as np

from cfnow._obj_functions import _obj_manhattan


class TestScriptBase(unittest.TestCase):
    def test__obj_manhattan_same_vectors_results_zero(self):
        vector_factual = np.array([1, 1])
        vector_cf = np.array([1, 1])
        result = _obj_manhattan(vector_factual, vector_cf)
        self.assertEqual(result, 0)

    def test__obj_manhattan_dif_vectors_calc(self):
        vector_factual = np.array([0, 1])
        vector_cf = np.array([1, 0])
        result = _obj_manhattan(vector_factual, vector_cf)
        self.assertEqual(result, 2)

    def test__obj_manhattan_result_is_never_zero(self):
        vector_factual = np.array([0, -1])
        vector_cf = np.array([-1, 0])
        result = _obj_manhattan(vector_factual, vector_cf)
        self.assertEqual(result, 2)


if __name__ == '__main__':
    unittest.main()
