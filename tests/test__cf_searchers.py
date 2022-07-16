import unittest
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd

from cfnow._cf_searchers import _create_change_matrix, _create_ohe_changes, _create_factual_changes, \
    _generate_random_changes_all_possibilities, _generate_random_changes_sample_possibilities, \
    _generate_random_changes, _random_generator_stop_conditions, _random_generator, \
    _greedy_generator_stop_conditions, _generate_greedy_changes, _greedy_generator


class TestScriptBase(unittest.TestCase):

    def test__create_change_matrix_only_cat_bin(self):
        factual = pd.Series({'a': 1, 'b': 0, 'c': 1})
        feat_types = {'a': 'cat', 'b': 'cat', 'c': 'cat'}
        ohe_indexes = []

        indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = \
            _create_change_matrix(factual, feat_types, ohe_indexes)

        self.assertListEqual(indexes_cat.tolist(), [0, 1, 2])
        self.assertListEqual(indexes_num, [])
        self.assertListEqual(arr_changes_cat_bin.tolist(), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        # For OHE we take all indexes because we need the row make reference to the column changed
        self.assertListEqual(arr_changes_cat_ohe.tolist(), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertListEqual(arr_changes_num.tolist(), [])

        self.assertIsInstance(indexes_cat, np.ndarray)
        self.assertIsInstance(arr_changes_cat_bin, np.ndarray)
        self.assertIsInstance(arr_changes_cat_ohe, np.ndarray)
        self.assertIsInstance(arr_changes_num, np.ndarray)

        self.assertTupleEqual(indexes_cat.shape, (3,))
        self.assertTupleEqual(arr_changes_cat_bin.shape, (3, 3))
        self.assertTupleEqual(arr_changes_cat_ohe.shape, (3, 3))
        self.assertTupleEqual(arr_changes_num.shape, (0, 3))

    def test__create_change_matrix_only_cat_ohe(self):
        factual = pd.Series({'d_0': 0, 'd_1': 0, 'd_2': 1, 'e_0': 1, 'e_1': 0})
        feat_types = {'d_0': 'cat', 'd_1': 'cat', 'd_2': 'cat', 'e_0': 'cat', 'e_1': 'cat'}
        ohe_indexes = [0, 1, 2, 3, 4]

        indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = \
            _create_change_matrix(factual, feat_types, ohe_indexes)

        self.assertListEqual(indexes_cat.tolist(), [0, 1, 2, 3, 4])
        self.assertListEqual(indexes_num, [])
        self.assertListEqual(arr_changes_cat_bin.tolist(), [])
        self.assertListEqual(arr_changes_cat_ohe.tolist(), [
            [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertListEqual(arr_changes_num.tolist(), [])

        self.assertIsInstance(indexes_cat, np.ndarray)
        self.assertIsInstance(arr_changes_cat_bin, np.ndarray)
        self.assertIsInstance(arr_changes_cat_ohe, np.ndarray)
        self.assertIsInstance(arr_changes_num, np.ndarray)

        self.assertTupleEqual(indexes_cat.shape, (5,))
        self.assertTupleEqual(arr_changes_cat_bin.shape, (0, 5))
        self.assertTupleEqual(arr_changes_cat_ohe.shape, (5, 5))
        self.assertTupleEqual(arr_changes_num.shape, (0, 5))

    def test__create_change_matrix_only_num(self):
        factual = pd.Series({'f': 0.5, 'g': 10})
        feat_types = {'f': 'num', 'g': 'num'}
        ohe_indexes = []

        indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = \
            _create_change_matrix(factual, feat_types, ohe_indexes)

        self.assertListEqual(indexes_cat.tolist(), [])
        self.assertListEqual(indexes_num, [0, 1])
        self.assertListEqual(arr_changes_cat_bin.tolist(), [])
        # For OHE we take all indexes because we need the row make reference to the column changed
        self.assertListEqual(arr_changes_cat_ohe.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertListEqual(arr_changes_num.tolist(), [[1.0, 0.0], [0.0, 1.0]])

        self.assertIsInstance(indexes_cat, np.ndarray)
        self.assertIsInstance(arr_changes_cat_bin, np.ndarray)
        self.assertIsInstance(arr_changes_cat_ohe, np.ndarray)
        self.assertIsInstance(arr_changes_num, np.ndarray)

        self.assertTupleEqual(indexes_cat.shape, (0,))
        self.assertTupleEqual(arr_changes_cat_bin.shape, (0, 2))
        self.assertTupleEqual(arr_changes_cat_ohe.shape, (2, 2))
        self.assertTupleEqual(arr_changes_num.shape, (2, 2))

    def test__create_change_matrix_num_bin_ohe(self):
        factual = pd.Series({'a': 1, 'b': 0, 'c': 1, 'd_0': 0, 'd_1': 0, 'd_2': 1,
                             'e_0': 1, 'e_1': 0, 'f': 0.5, 'g': 10})
        feat_types = {'a': 'cat', 'b': 'cat', 'c': 'cat', 'd_0': 'cat', 'd_1': 'cat', 'd_2': 'cat',
                      'e_0': 'cat', 'e_1': 'cat', 'f': 'num', 'g': 'num'}
        ohe_indexes = [3, 4, 5, 6, 7]

        indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = \
            _create_change_matrix(factual, feat_types, ohe_indexes)

        self.assertListEqual(indexes_cat.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(indexes_num, [8, 9])

        self.assertListEqual(arr_changes_cat_bin.tolist(), [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # We take all indexes for OHE because we need the row idx to correspond to the column change idx
        self.assertListEqual(arr_changes_cat_ohe.tolist(), [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        self.assertListEqual(arr_changes_num.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        self.assertIsInstance(indexes_cat, np.ndarray)
        self.assertIsInstance(arr_changes_cat_bin, np.ndarray)
        self.assertIsInstance(arr_changes_cat_ohe, np.ndarray)
        self.assertIsInstance(arr_changes_num, np.ndarray)

        self.assertTupleEqual(indexes_cat.shape, (8,))
        self.assertTupleEqual(arr_changes_cat_bin.shape, (3, 10))
        self.assertTupleEqual(arr_changes_cat_ohe.shape, (10, 10))
        self.assertTupleEqual(arr_changes_num.shape, (2, 10))

    def test__create_ohe_changes_only_ohe(self):
        # 3 OHE encoded features with 3 values
        cf_try = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        ohe_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list], [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        self.assertListEqual(changes_cat_ohe.tolist(), [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertIsInstance(changes_cat_ohe_list[0], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[1], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[2], np.ndarray)

    def test__create_ohe_changes_num_bin_ohe(self):
        # 2 numerical, 2 binary and 3 OHE encoded features with 3 values
        cf_try = np.array([10, 50, 0, 1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        ohe_list = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list], [
            [[0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0., 0., 0., 0., -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0., 0., 0., 0., -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0., 0., 0., 0., 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
             [0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]],

            [[0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
             [0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
             [0., 0., 0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        self.assertListEqual(changes_cat_ohe.tolist(), [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertIsInstance(changes_cat_ohe_list[0], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[1], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[2], np.ndarray)

    def test__create_ohe_changes_ohe_num_bin(self):
        # 2 numerical, 2 binary and 3 OHE encoded features with 3 values
        cf_try = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10, 50, 0, 1])
        ohe_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list], [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.],
             [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.],
             [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.]],

            [[0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.]],

            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0.]]])

        self.assertListEqual(changes_cat_ohe.tolist(), [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertIsInstance(changes_cat_ohe_list[0], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[1], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[2], np.ndarray)

    def test__create_ohe_changes_num_ohe_bin(self):
        # 2 numerical, 2 binary and 3 OHE encoded features with 3 values
        cf_try = np.array([10, 50, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 1])
        ohe_list = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list], [
            [[0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.],
             [0., 0., -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.],
             [0., 0., -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.]],

            [[0., 0., 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0., 0.],
             [0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.],
             [0., 0., 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0., 0.]],

            [[0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0., 0.],
             [0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0., 0.],
             [0., 0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0., 0.]]])

        self.assertListEqual(changes_cat_ohe.tolist(), [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertIsInstance(changes_cat_ohe_list[0], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[1], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[2], np.ndarray)

    def test__create_ohe_changes_ohe_num_ohe_bin_ohe(self):
        # 2 numerical, 2 binary and 3 OHE encoded features with 3 values
        cf_try = np.array([1.0, 0.0, 0.0, 10, 50, 0.0, 1.0, 0.0, 0, 1, 0.0, 0.0, 1.0])
        ohe_list = [[0, 1, 2], [5, 6, 7], [10, 11, 12]]
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list], [
            [[0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0],
             [-1.0, 1.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0],
             [-1.0, 0.0, 1.0, 0., 0., 0.0, 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0., 0., 1.0, -1.0, 0.0, 0., 0., 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0., 0., 0.0, -1.0, 1.0, 0., 0., 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0., 0., 1.0, 0.0, -1.0],
             [0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0., 0., 0.0, 1.0, -1.0],
             [0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0, 0., 0., 0.0, 0.0, 0.0]]])

        self.assertListEqual(changes_cat_ohe.tolist(), [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertIsInstance(changes_cat_ohe_list[0], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[1], np.ndarray)
        self.assertIsInstance(changes_cat_ohe_list[2], np.ndarray)

    def test__create_ohe_changes_no_ohe(self):
        # 2 numerical, 2 binary and 3 OHE encoded features with 3 values
        cf_try = np.array([10, 50, 0, 1])
        ohe_list = []
        arr_changes_cat_ohe = np.array([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])

        changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_ohe_list, [])
        self.assertListEqual(changes_cat_ohe, [])

    def test__create_factual_changes_no_momentum_num(self):
        cf_try = np.array([10, 50])
        ohe_list = []
        ft_change_factor = 0.5
        add_momentum = False
        arr_changes_num = np.array([[1.0, 0.0], [0.0, 1.0]])
        arr_changes_cat_bin = np.array([]).reshape((0, 2))
        arr_changes_cat_ohe = np.array([[1.0, 0.0], [0.0, 1.0]])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        a = 1