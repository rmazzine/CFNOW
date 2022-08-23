import unittest
from collections import deque
from unittest.mock import patch, MagicMock, call

import datetime

import numpy as np
import pandas as pd

from cfnow._cf_searchers import _create_change_matrix, _create_ohe_changes, _create_factual_changes, _count_subarray, \
    _replace_ohe_placeholders, _replace_num_placeholders, _generate_random_changes_all_possibilities, \
    _create_random_changes, _generate_random_changes_sample_possibilities, _calc_num_possible_changes, \
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
        cf_try = np.array([10, -50])
        ohe_list = []
        ft_change_factor = 0.5
        add_momentum = 0
        arr_changes_num = np.array([[1.0, 0.0], [0.0, 1.0]])
        arr_changes_cat_bin = np.array([]).reshape((0, 2))
        arr_changes_cat_ohe = np.array([[1.0, 0.0], [0.0, 1.0]])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [])
        self.assertListEqual(changes_cat_ohe_list, [])
        self.assertListEqual(changes_cat_ohe, [])
        self.assertListEqual(changes_num_up.tolist(), [[5.0, 0.0], [0.0, -25.0]])
        self.assertListEqual(changes_num_down.tolist(), [[-5.0, 0.0], [0.0, 25.0]])

        self.assertTupleEqual(changes_cat_bin.shape, (0, 2))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_with_momentum_num(self):
        cf_try = np.array([10, -50])
        ohe_list = []
        ft_change_factor = 0.5
        add_momentum = 1
        arr_changes_num = np.array([[1.0, 0.0], [0.0, 1.0]])
        arr_changes_cat_bin = np.array([]).reshape((0, 2))
        arr_changes_cat_ohe = np.array([[1.0, 0.0], [0.0, 1.0]])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [])
        self.assertListEqual(changes_cat_ohe_list, [])
        self.assertListEqual(changes_cat_ohe, [])
        self.assertListEqual(changes_num_up.tolist(), [[6.0, 0.0], [0.0, -26.0]])
        self.assertListEqual(changes_num_down.tolist(), [[-6.0, 0.0], [0.0, 26.0]])

        self.assertTupleEqual(changes_cat_bin.shape, (0, 2))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_no_momentum_bin(self):
        cf_try = np.array([1, 0])
        ohe_list = []
        ft_change_factor = 0.5
        add_momentum = 0
        arr_changes_num = np.array([]).reshape((0, 2))
        arr_changes_cat_bin = np.array([[1.0, 0.0], [0.0, 1.0]])
        arr_changes_cat_ohe = np.array([[1.0, 0.0], [0.0, 1.0]])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [[-1, 0], [0, 1]])
        self.assertListEqual(changes_cat_ohe_list, [])
        self.assertListEqual(changes_cat_ohe, [])
        self.assertListEqual(changes_num_up.tolist(), [])
        self.assertListEqual(changes_num_down.tolist(), [])

        self.assertTupleEqual(changes_num_up.shape, (0, 2))
        self.assertTupleEqual(changes_num_down.shape, (0, 2))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_with_momentum_bin(self):
        cf_try = np.array([1, 0])
        ohe_list = []
        ft_change_factor = 0.5
        add_momentum = 1
        arr_changes_num = np.array([]).reshape((0, 2))
        arr_changes_cat_bin = np.array([[1.0, 0.0], [0.0, 1.0]])
        arr_changes_cat_ohe = np.array([[1.0, 0.0], [0.0, 1.0]])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum, arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [[-1, 0], [0, 1]])
        self.assertListEqual(changes_cat_ohe_list, [])
        self.assertListEqual(changes_cat_ohe, [])
        self.assertListEqual(changes_num_up.tolist(), [])
        self.assertListEqual(changes_num_down.tolist(), [])

        self.assertTupleEqual(changes_num_up.shape, (0, 2))
        self.assertTupleEqual(changes_num_down.shape, (0, 2))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_no_momentum_ohe(self):
        cf_try = np.array([1, 0, 0, 0, 1, 0])
        ohe_list = [[0, 1, 2], [3, 4, 5]]
        ft_change_factor = 0.5
        add_momentum = 0
        arr_changes_num = np.array([]).reshape((0, 6))
        arr_changes_cat_bin = np.array([]).reshape((0, 6))
        arr_changes_cat_ohe = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [])
        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list],
                             [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],

                              [[0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]]])
        self.assertListEqual(changes_cat_ohe.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        self.assertListEqual(changes_num_up.tolist(), [])
        self.assertListEqual(changes_num_down.tolist(), [])

        self.assertTupleEqual(changes_cat_bin.shape, (0, 6))
        self.assertTupleEqual(changes_num_up.shape, (0, 6))
        self.assertTupleEqual(changes_num_down.shape, (0, 6))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_with_momentum_ohe(self):
        cf_try = np.array([1, 0, 0, 0, 1, 0])
        ohe_list = [[0, 1, 2], [3, 4, 5]]
        ft_change_factor = 0.5
        add_momentum = 1
        arr_changes_num = np.array([]).reshape((0, 6))
        arr_changes_cat_bin = np.array([]).reshape((0, 6))
        arr_changes_cat_ohe = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(), [])
        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list],
                             [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],

                              [[0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]]])
        self.assertListEqual(changes_cat_ohe.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        self.assertListEqual(changes_num_up.tolist(), [])
        self.assertListEqual(changes_num_down.tolist(), [])

        self.assertTupleEqual(changes_cat_bin.shape, (0, 6))
        self.assertTupleEqual(changes_num_up.shape, (0, 6))
        self.assertTupleEqual(changes_num_down.shape, (0, 6))

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_no_momentum_num_ohe_bin_ohe(self):
        # [num, num, ohe1, ohe1, ohe1, bin, bin, ohe2, ohe2, ohe2]
        cf_try = np.array([-50, 10, 1, 0, 0, 1, 0, 0, 1, 0])
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ft_change_factor = 0.5
        add_momentum = 0
        arr_changes_num = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        arr_changes_cat_bin = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        arr_changes_cat_ohe = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list],
                             [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

                              [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]]])
        self.assertListEqual(changes_cat_ohe.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        self.assertListEqual(changes_num_up.tolist(), [[-25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [-0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertListEqual(changes_num_down.tolist(), [[25.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                                         [0.0, -5.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__create_factual_changes_with_momentum_num_ohe_bin_ohe(self):
        # [num, num, ohe1, ohe1, ohe1, bin, bin, ohe2, ohe2, ohe2]
        cf_try = np.array([-50, 10, 1, 0, 0, 1, 0, 0, 1, 0])
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ft_change_factor = 0.5
        add_momentum = 1
        arr_changes_num = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        arr_changes_cat_bin = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        arr_changes_cat_ohe = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                                    arr_changes_cat_bin, arr_changes_cat_ohe)

        self.assertListEqual(changes_cat_bin.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        self.assertListEqual([cl.tolist() for cl in changes_cat_ohe_list],
                             [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

                              [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]]])
        self.assertListEqual(changes_cat_ohe.tolist(),
                             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        self.assertListEqual(changes_num_up.tolist(), [[-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                       [-0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertListEqual(changes_num_down.tolist(), [[26.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                                         [0.0, -6.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])

        self.assertIsInstance(changes_cat_bin, np.ndarray)
        self.assertIsInstance(changes_num_up, np.ndarray)
        self.assertIsInstance(changes_num_down, np.ndarray)

    def test__count_subarray_example(self):
        sa = [[0], [0, 0, 0, 0]]

        result_sa = _count_subarray(sa)

        self.assertEqual(result_sa, 5)

    def test__replace_ohe_placeholders_bin(self):
        update_placeholder = [[0, 1]]
        ohe_placeholders = []
        ohe_placeholder_to_change_idx = {}
        pc_idx_ohe = []

        update_placeholder = _replace_ohe_placeholders(update_placeholder, ohe_placeholders,
                                                       ohe_placeholder_to_change_idx, pc_idx_ohe)

        self.assertListEqual(update_placeholder, [[0, 1]])

    def test__replace_ohe_placeholders_num(self):
        update_placeholder = [['num_0', 'num_1']]
        ohe_placeholders = []
        ohe_placeholder_to_change_idx = {}
        pc_idx_ohe = []

        update_placeholder = _replace_ohe_placeholders(update_placeholder, ohe_placeholders,
                                                       ohe_placeholder_to_change_idx, pc_idx_ohe)

        self.assertListEqual(update_placeholder, [['num_0', 'num_1']])

    def test__replace_ohe_placeholders_ohe(self):
        update_placeholder = [['ohe_0', 'ohe_1']]
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        pc_idx_ohe = [0, 1, 2, 3, 4, 5]

        update_placeholder = _replace_ohe_placeholders(update_placeholder, ohe_placeholders,
                                                       ohe_placeholder_to_change_idx, pc_idx_ohe)

        self.assertListEqual(update_placeholder, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4],
                                                  [1, 5], [2, 3], [2, 4], [2, 5]])

    def test__replace_ohe_placeholders_bin_ohe_num(self):
        update_placeholder = [[0, 1], [0, 'num_0'], [0, 'num_1'], [0, 'ohe_0'], [0, 'ohe_1'], [1, 'num_0'],
                              [1, 'num_1'], [1, 'ohe_0'], [1, 'ohe_1'], ['num_0', 'num_1'], ['num_0', 'ohe_0'],
                              ['num_0', 'ohe_1'], ['num_1', 'ohe_0'], ['num_1', 'ohe_1'], ['ohe_0', 'ohe_1']]
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]

        update_placeholder = _replace_ohe_placeholders(update_placeholder, ohe_placeholders,
                                                       ohe_placeholder_to_change_idx, pc_idx_ohe)

        self.assertListEqual(update_placeholder,
                             [[0, 1], [0, 'num_0'], [0, 'num_1'], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                              [1, 'num_0'], [1, 'num_1'], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                              ['num_0', 'num_1'], ['num_0', 2], ['num_0', 3], ['num_0', 4], ['num_0', 5],
                              ['num_0', 6], ['num_0', 7], ['num_1', 2], ['num_1', 3], ['num_1', 4], ['num_1', 5],
                              ['num_1', 6], ['num_1', 7], [2, 5], [2, 6], [2, 7], [3, 5], [3, 6], [3, 7],
                              [4, 5], [4, 6], [4, 7]])

    def test__replace_num_placeholders_bin_ohe_num(self):
        update_placeholder = [[0, 1], [0, 'num_0'], [0, 'num_1'], [0, 'ohe_0'], [0, 'ohe_1'], [1, 'num_0'],
                              [1, 'num_1'], [1, 'ohe_0'], [1, 'ohe_1'], ['num_0', 'num_1'], ['num_0', 'ohe_0'],
                              ['num_0', 'ohe_1'], ['num_1', 'ohe_0'], ['num_1', 'ohe_1'], ['ohe_0', 'ohe_1']]
        num_placeholders = ['num_0', 'num_1']
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        update_placeholder = _replace_num_placeholders(update_placeholder, num_placeholders,
                                                       num_placeholder_to_change_idx, pc_idx_nup, pc_idx_ndw)

        self.assertListEqual(update_placeholder, [[0, 1], [0, 8], [0, 10], [0, 9], [0, 11], [0, 'ohe_0'], [0, 'ohe_1'],
                                                  [1, 8], [1, 10], [1, 9], [1, 11], [1, 'ohe_0'], [1, 'ohe_1'], [8, 9],
                                                  [8, 11], [10, 9], [10, 11], [8, 'ohe_0'], [10, 'ohe_0'],
                                                  [8, 'ohe_1'], [10, 'ohe_1'], [9, 'ohe_0'], [11, 'ohe_0'],
                                                  [9, 'ohe_1'], [11, 'ohe_1'], ['ohe_0', 'ohe_1']])

    def test__replace_num_placeholders_bin(self):
        update_placeholder = [[0, 1]]
        num_placeholders = []
        num_placeholder_to_change_idx = {}
        pc_idx_nup = []
        pc_idx_ndw = []

        update_placeholder = _replace_num_placeholders(update_placeholder, num_placeholders,
                                                       num_placeholder_to_change_idx, pc_idx_nup, pc_idx_ndw)

        self.assertListEqual(update_placeholder, [[0, 1]])

    def test__replace_num_placeholders_ohe(self):
        update_placeholder = [['ohe_0', 'ohe_1']]
        num_placeholders = []
        num_placeholder_to_change_idx = {}
        pc_idx_nup = []
        pc_idx_ndw = []

        update_placeholder = _replace_num_placeholders(update_placeholder, num_placeholders,
                                                       num_placeholder_to_change_idx, pc_idx_nup, pc_idx_ndw)

        self.assertListEqual(update_placeholder, [['ohe_0', 'ohe_1']])

    def test__replace_num_placeholders_num(self):
        update_placeholder = [['num_0', 'num_1']]
        num_placeholders = ['num_0', 'num_1']
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_nup = [0, 1]
        pc_idx_ndw = [2, 3]

        update_placeholder = _replace_num_placeholders(update_placeholder, num_placeholders,
                                                       num_placeholder_to_change_idx, pc_idx_nup, pc_idx_ndw)

        self.assertListEqual(update_placeholder, [[0, 1], [0, 3], [2, 1], [2, 3]])

    def test__generate_random_changes_all_possibilities_bin_ohe_num(self):
        n_changes = 2
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = [0, 1, 'num_0', 'num_1', 'ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}

        changes_idx = _generate_random_changes_all_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx)

        self.assertListEqual(changes_idx, [[0, 1], [0, 8], [0, 10], [0, 9], [0, 11], [0, 2], [0, 3], [0, 4], [0, 5],
                                           [0, 6], [0, 7], [1, 8], [1, 10], [1, 9], [1, 11], [1, 2], [1, 3], [1, 4],
                                           [1, 5], [1, 6], [1, 7], [8, 9], [8, 11], [10, 9], [10, 11], [8, 2], [10, 2],
                                           [8, 3], [10, 3], [8, 4], [10, 4], [8, 5], [10, 5], [8, 6], [10, 6], [8, 7],
                                           [10, 7], [9, 2], [11, 2], [9, 3], [11, 3], [9, 4], [11, 4], [9, 5], [11, 5],
                                           [9, 6], [11, 6], [9, 7], [11, 7], [2, 5], [2, 6], [2, 7], [3, 5], [3, 6],
                                           [3, 7], [4, 5], [4, 6], [4, 7]])

    def test__generate_random_changes_all_possibilities_bin(self):
        n_changes = 2
        pc_idx_ohe = []
        pc_idx_nup = []
        pc_idx_ndw = []
        num_placeholders = []
        ohe_placeholders = []
        change_feat_options = [0, 1]
        ohe_placeholder_to_change_idx = {}
        num_placeholder_to_change_idx = {}

        changes_idx = _generate_random_changes_all_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx)

        self.assertListEqual(changes_idx, [[0, 1]])

    def test__generate_random_changes_all_possibilities_ohe(self):
        n_changes = 2
        pc_idx_ohe = [0, 1, 2, 3, 4, 5]
        pc_idx_nup = []
        pc_idx_ndw = []
        num_placeholders = []
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {}

        changes_idx = _generate_random_changes_all_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx)

        self.assertListEqual(changes_idx, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]])

    def test__generate_random_changes_all_possibilities_num(self):
        n_changes = 2
        pc_idx_ohe = []
        pc_idx_nup = [0, 1]
        pc_idx_ndw = [2, 3]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = []
        change_feat_options = ['num_0', 'num_1']
        ohe_placeholder_to_change_idx = {}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}

        changes_idx = _generate_random_changes_all_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx)

        self.assertListEqual(changes_idx, [[0, 1], [0, 3], [2, 1], [2, 3]])

    def test__create_random_changes_bin(self):
        sample_features = [0, 1]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {0, 1})

    @patch('cfnow._cf_searchers.np')
    def test__create_random_changes_ohe(self, mock_np):
        sample_features = ['ohe_0', 'ohe_1']
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        mock_np.random.choice.side_effect = [[True], [False]]

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {False, True})
        mock_np.random.choice.assert_has_calls([call([2, 3, 4], 1), call([5, 6, 7], 1)])

    @patch('cfnow._cf_searchers.np')
    def test__create_random_changes_num(self, mock_np):
        sample_features = ['num_0', 'num_1']
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        mock_np.random.choice.side_effect = [True, False]

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {False, True})
        mock_np.random.choice.assert_has_calls([call([8, 10]), call([9, 11])])

    @patch('cfnow._cf_searchers.np')
    def test__create_random_changes_bin_ohe(self, mock_np):
        sample_features = [0, 'ohe_0']
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        mock_np.random.choice.return_value = [True]

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {0, True})
        mock_np.random.choice.assert_called_once_with([2, 3, 4], 1)

    @patch('cfnow._cf_searchers.np')
    def test__create_random_changes_bin_num(self, mock_np):
        sample_features = [0, 'num_0']
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        mock_np.random.choice.return_value = True

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {0, True})
        mock_np.random.choice.assert_called_once_with([8, 10])

    @patch('cfnow._cf_searchers.np')
    def test__create_random_changes_ohe_num(self, mock_np):
        sample_features = ['ohe_0', 'num_0']
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]

        mock_np.random.choice.side_effect = [[True], False]

        change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        self.assertSetEqual(change_idx_row, {False, True})
        mock_np.random.choice.assert_has_calls([call([2, 3, 4], 1), call([8, 10])])

    @patch('cfnow._cf_searchers._create_random_changes')
    @patch('cfnow._cf_searchers.np')
    def test__generate_random_changes_sample_possibilities_example(self, mock_np, mock_create_random_changes):
        n_changes = 2
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = [0, 1, 'num_0', 'num_1', 'ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        threshold_changes = 2

        mock_np.random.choice.side_effect = [[0, 'num_0', 'ohe_0'], [1, 'num_1', 'ohe_1']]
        mock_create_random_changes.side_effect = [[0, 1, 2], [2, 3, 4]]

        changes_idx = _generate_random_changes_sample_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx, threshold_changes)

        mock_np.random.choice.assert_called()
        mock_create_random_changes.assert_called()

        self.assertListEqual(changes_idx, [[0, 1, 2], [2, 3, 4]])

    @patch('cfnow._cf_searchers._create_random_changes')
    @patch('cfnow._cf_searchers.np')
    def test__generate_random_changes_sample_possibilities_two_same_ohe(self, mock_np, mock_create_random_changes):
        n_changes = 2
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = [0, 1, 'num_0', 'num_1', 'ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        threshold_changes = 2

        mock_np.random.choice.side_effect = [[0, 'num_0', 'ohe_0'], [1, 'ohe_1', 'ohe_1'], [1, 'ohe_1', 'ohe_1'],
                                             [1, 'ohe_1', 'ohe_1'], [1, 'ohe_1', 'ohe_1']]
        mock_create_random_changes.side_effect = [[0, 1, 2]]

        changes_idx = _generate_random_changes_sample_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx, threshold_changes)

        mock_np.random.choice.assert_called()
        mock_create_random_changes.assert_called_once()

        self.assertListEqual(changes_idx, [[0, 1, 2]])

    @patch('cfnow._cf_searchers._create_random_changes')
    @patch('cfnow._cf_searchers.np')
    def test__generate_random_changes_sample_possibilities_two_same_bin(self, mock_np, mock_create_random_changes):
        n_changes = 2
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = [0, 1, 'num_0', 'num_1', 'ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        threshold_changes = 2

        mock_np.random.choice.side_effect = [[0, 'num_0', 'ohe_0'], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

        mock_create_random_changes.side_effect = [[0, 1, 2]]

        changes_idx = _generate_random_changes_sample_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx, threshold_changes)

        mock_np.random.choice.assert_called()
        mock_create_random_changes.assert_called_once()

        self.assertListEqual(changes_idx, [[0, 1, 2]])

    @patch('cfnow._cf_searchers._create_random_changes')
    @patch('cfnow._cf_searchers.np')
    def test__generate_random_changes_sample_possibilities_two_same_num(self, mock_np, mock_create_random_changes):
        n_changes = 2
        pc_idx_ohe = [2, 3, 4, 5, 6, 7]
        pc_idx_nup = [8, 9]
        pc_idx_ndw = [10, 11]
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = [0, 1, 'num_0', 'num_1', 'ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        num_placeholder_to_change_idx = {'num_0': 0, 'num_1': 1}
        threshold_changes = 2

        mock_np.random.choice.side_effect = [[0, 'num_0', 'ohe_0'], [1, 'num_0', 'num_0']]

        mock_create_random_changes.side_effect = [[0, 1, 2], [2, 3, 4]]

        changes_idx = _generate_random_changes_sample_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx, threshold_changes)

        mock_np.random.choice.assert_called()
        mock_create_random_changes.assert_called()

        self.assertListEqual(changes_idx, [[0, 1, 2], [2, 3, 4]])

    def test__calc_num_possible_changes_bin_ohe_num(self):
        num_placeholders = ['num_0']
        ohe_placeholders = ['ohe_0']
        change_feat_options = [0, 1, 'ohe_0', 'num_0']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3]}
        threshold_changes = 1000
        n_changes = 2

        corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                           ohe_placeholder_to_change_idx, threshold_changes, n_changes)

        self.assertEqual(corrected_num_changes, 17)

    def test__calc_num_possible_changes_bin(self):
        num_placeholders = []
        ohe_placeholders = []
        change_feat_options = [0, 1]
        ohe_placeholder_to_change_idx = {}
        threshold_changes = 1000
        n_changes = 2

        corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                           ohe_placeholder_to_change_idx, threshold_changes, n_changes)

        self.assertEqual(corrected_num_changes, 1)

    def test__calc_num_possible_changes_ohe(self):
        num_placeholders = []
        ohe_placeholders = ['ohe_0', 'ohe_1']
        change_feat_options = ['ohe_0', 'ohe_1']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3], 'ohe_1': [3, 6]}
        threshold_changes = 1000
        n_changes = 2

        corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                           ohe_placeholder_to_change_idx, threshold_changes, n_changes)

        self.assertEqual(corrected_num_changes, 9)

    def test__calc_num_possible_changes_num(self):
        num_placeholders = ['num_0', 'num_1']
        ohe_placeholders = []
        change_feat_options = ['num_0', 'num_1']
        ohe_placeholder_to_change_idx = {}
        threshold_changes = 1000
        n_changes = 2

        corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                           ohe_placeholder_to_change_idx, threshold_changes, n_changes)

        self.assertEqual(corrected_num_changes, 4)

    def test__calc_num_possible_changes_above_threshold(self):
        num_placeholders = ['num_0']
        ohe_placeholders = ['ohe_0']
        change_feat_options = [0, 1, 'ohe_0', 'num_0']
        ohe_placeholder_to_change_idx = {'ohe_0': [0, 3]}
        threshold_changes = 1
        n_changes = 2

        corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                           ohe_placeholder_to_change_idx, threshold_changes, n_changes)

        self.assertEqual(corrected_num_changes, 2)

    def test__generate_random_changes_num_ohe_bin_ohe(self):
        changes_cat_bin = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([[-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        changes_num_down = np.array([[26.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                     [0.0, -6.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        n_changes = 2
        threshold_changes = 1000

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                                                         [-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertListEqual(changes_idx, [[0, 1], [0, 8], [0, 10], [0, 9], [0, 11], [0, 2], [0, 3], [0, 4], [0, 5],
                                           [0, 6], [0, 7], [1, 8], [1, 10], [1, 9], [1, 11], [1, 2], [1, 3], [1, 4],
                                           [1, 5], [1, 6], [1, 7], [8, 9], [8, 11], [10, 9], [10, 11], [8, 2], [10, 2],
                                           [8, 3], [10, 3], [8, 4], [10, 4], [8, 5], [10, 5], [8, 6], [10, 6], [8, 7],
                                           [10, 7], [9, 2], [11, 2], [9, 3], [11, 3], [9, 4], [11, 4], [9, 5], [11, 5],
                                           [9, 6], [11, 6], [9, 7], [11, 7], [2, 5], [2, 6], [2, 7], [3, 5], [3, 6],
                                           [3, 7], [4, 5], [4, 6], [4, 7]])

    def test__generate_random_changes_num(self):
        changes_cat_bin = np.array([]).reshape((0, 2))
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([[-26.0, 0.0],
                                   [-0.0, 6.0]])
        changes_num_down = np.array([[26.0, -0.0],
                                     [0.0, -6.0]])
        ohe_list = []
        n_changes = 2
        threshold_changes = 1000

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[-26.0, 0.0],
                                                         [-0.0, 6.0],
                                                         [26.0, -0.0],
                                                         [0.0, -6.0]])

        self.assertListEqual(changes_idx, [[0, 1], [0, 3], [2, 1], [2, 3]])

    def test__generate_random_changes_ohe(self):
        changes_cat_bin = np.array([])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        ohe_list = [[0, 1, 2], [3, 4, 5]]
        n_changes = 2
        threshold_changes = 1000

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                         [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])

        self.assertListEqual(changes_idx, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]])

    def test__generate_random_changes_bin(self):
        changes_cat_bin = np.array([[-1.0, 0.0],
                                    [0.0, 1.0]])
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        ohe_list = []
        n_changes = 2
        threshold_changes = 1000

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[-1.0, 0.0],
                                                         [0.0, 1.0]])

        self.assertListEqual(changes_idx, [[0, 1]])

    @patch('cfnow._cf_searchers._generate_random_changes_sample_possibilities')
    def test__generate_random_changes_num_ohe_bin_ohe_above_threshold(
            self, mock_generate_random_changes_sample_possibilities):
        changes_cat_bin = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([[-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        changes_num_down = np.array([[26.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                     [0.0, -6.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        n_changes = 2
        threshold_changes = 1

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                                                         [-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        mock_generate_random_changes_sample_possibilities.assert_called_once()

    @patch('cfnow._cf_searchers._generate_random_changes_sample_possibilities')
    def test__generate_random_changes_num_above_threshold(self, mock_generate_random_changes_sample_possibilities):
        changes_cat_bin = np.array([]).reshape((0, 2))
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([[-26.0, 0.0],
                                   [-0.0, 6.0]])
        changes_num_down = np.array([[26.0, -0.0],
                                     [0.0, -6.0]])
        ohe_list = []
        n_changes = 2
        threshold_changes = 1

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[-26.0, 0.0],
                                                         [-0.0, 6.0],
                                                         [26.0, -0.0],
                                                         [0.0, -6.0]])

        mock_generate_random_changes_sample_possibilities.assert_called_once()

    @patch('cfnow._cf_searchers._generate_random_changes_sample_possibilities')
    def test__generate_random_changes_ohe_above_threshold(self, mock_generate_random_changes_sample_possibilities):
        changes_cat_bin = np.array([])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        ohe_list = [[0, 1, 2], [3, 4, 5]]
        n_changes = 2
        threshold_changes = 1

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                         [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                         [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])

        mock_generate_random_changes_sample_possibilities.assert_called_once()

    @patch('cfnow._cf_searchers._generate_random_changes_sample_possibilities')
    def test__generate_random_changes_bin_above_threshold(self, mock_generate_random_changes_sample_possibilities):
        changes_cat_bin = np.array([[-1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0]])
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        ohe_list = []
        n_changes = 2
        threshold_changes = 1

        possible_changes, changes_idx = _generate_random_changes(
            changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list, n_changes, threshold_changes)

        self.assertListEqual(possible_changes.tolist(), [[-1.0, 0.0, 0.0],
                                                         [0.0, 1.0, 0.0],
                                                         [0.0, 0.0, 1.0]])

        mock_generate_random_changes_sample_possibilities.assert_called_once()

    @patch('cfnow._cf_searchers.datetime')
    def test__random_generator_stop_conditions_continue_loop(self, mock_datetime):
        iterations = 1
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _random_generator_stop_conditions(iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf)

        self.assertTrue(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__random_generator_stop_conditions_cf_found(self, mock_datetime):
        iterations = 1
        cf_unique = [[1, 2, 3]]
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _random_generator_stop_conditions(iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__random_generator_stop_conditions_max_iterations_reached(self, mock_datetime):
        iterations = 100
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _random_generator_stop_conditions(iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__random_generator_stop_conditions_timeout(self, mock_datetime):
        iterations = 1
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 20, 10)

        result = _random_generator_stop_conditions(iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_example(self, mock_random_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)
        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # No CF can be equal to the factual
        self.assertTrue(sum([np.array_equal(cf, factual_np) for cf in cf_try]) == 0)

        # Since we have 10 results with probability above 0.5 we expect 10 results
        self.assertEqual(len(cf_try), 10)

        # Verify if iterations variable was incremented
        self.assertEqual(mock_random_generator_stop_conditions.call_args_list[1][1]['iterations'], 2)

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_best_cf(self, mock_random_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns two counterfactuals
            # [-50, 10, 0, 1, 0, 1, 0, 0, 1, 0] => p = 0.51
            # [-50, 10, 1, 0, 0, 1, 0, 1, 0, 0] => p = 0.81

            cf_tries = x
            if type(x) == pd.DataFrame:
                cf_tries = x.to_numpy()

            out_result = []
            for row in cf_tries:
                if (row != np.array([-50, 10, 0, 1, 0, 1, 0, 0, 1, 0])).sum() == 0:
                    out_result.append(0.51)
                elif (row != np.array([-50, 10, 1, 0, 0, 1, 0, 1, 0, 0])).sum() == 0:
                    out_result.append(0.81)
                else:
                    out_result.append(0.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if the first result is the best CF
        self.assertListEqual(cf_try[0].tolist(), [-50, 10, 1, 0, 0, 1, 0, 1, 0, 0])

    @patch('cfnow._cf_searchers._create_factual_changes')
    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_momentum_increase(
            self, mock_random_generator_stop_conditions, mock_create_factual_changes):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        # This function will run one full sequence of feature modification loops, then, in the second sequence
        # In the first change it will return a 1.0 prediction, this is done because this will allow the
        # momentum increment.
        def _mp1c_side_effect_function(cf_tries):
            if mp1c.call_count > 5:
                return np.array([1.0]*len(cf_tries))
            return np.array([0.0]*len(cf_tries))

        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, True, False]
        mock_create_factual_changes.side_effect = lambda *args: _create_factual_changes(*args)

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if momentum was incremented (equal to 1) in the last iteration
        self.assertEqual(mock_create_factual_changes.call_args_list[-1][0][3], 1)

    @patch('cfnow._cf_searchers._generate_random_changes')
    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_threshold_changes(
            self, mock_random_generator_stop_conditions, mock_generate_random_changes):
        finder_strategy = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = []
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        # Verify threshold for tabular data
        cf_data_type = 'tabular'
        mock_random_generator_stop_conditions.side_effect = [True, False]
        mock_generate_random_changes.side_effect = lambda *args: _generate_random_changes(*args)
        cf_try_tabular_threshold = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)
        # Assert the threshold is 1000
        self.assertEqual(mock_generate_random_changes.call_args_list[0][0][6], 1000)

        # Verify threshold for image data
        cf_data_type = 'image'
        mock_random_generator_stop_conditions.side_effect = [True, False]
        mock_generate_random_changes.side_effect = lambda *args: _generate_random_changes(*args)
        cf_try_tabular_threshold = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)
        # Assert the threshold is 1000
        self.assertEqual(mock_generate_random_changes.call_args_list[1][0][6], 1000)

        # Verify threshold for text data
        cf_data_type = 'text'
        mock_random_generator_stop_conditions.side_effect = [True, False]
        mock_generate_random_changes.side_effect = lambda *args: _generate_random_changes(*args)
        cf_try_tabular_threshold = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)
        # Assert the threshold is 1000
        self.assertEqual(mock_generate_random_changes.call_args_list[2][0][6], 1000)

    @patch('cfnow._cf_searchers._generate_random_changes')
    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_no_possible_changes(
            self, mock_random_generator_stop_conditions, mock_generate_random_changes):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = []
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        mock_generate_random_changes.return_value = [
            np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                       [-26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
            []]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # In this case, as there are no changes, the output CF must be an empty array
        self.assertTrue(len(cf_try) == 0)

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_all_features_tabu(
            self, mock_random_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = [[0], [1], [2, 3, 4], [5], [6], [7, 8, 9]]
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # In this case, as there are no changes (since all features are in the Tabu list),
        # the output must be an empty list
        self.assertTrue(len(cf_try) == 0)

    @patch('cfnow._cf_searchers.logging')
    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_verbose(self, mock_random_generator_stop_conditions, mock_logging):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = True

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        mock_logging.log.assert_called()

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_continue_loop(self, mock_datetime):
        iterations = 1
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 1.0
        increase_threshold = 0.0
        activate_tabu = False

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertTrue(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_cf_found(self, mock_datetime):
        iterations = 1
        cf_unique = [[0, 1]]
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 1.0
        increase_threshold = 0.0
        activate_tabu = False

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_iterations(self, mock_datetime):
        iterations = 100
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 1.0
        increase_threshold = 0.0
        activate_tabu = False

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_score_increase_no_tabu(self, mock_datetime):
        iterations = 0
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 0.0
        increase_threshold = 1.0
        activate_tabu = False

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertFalse(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_score_increase_with_tabu(self, mock_datetime):
        iterations = 0
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 0.0
        increase_threshold = 1.0
        activate_tabu = True

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 10, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertTrue(result)

    @patch('cfnow._cf_searchers.datetime')
    def test__greedy_generator_stop_conditions_score_timeout(self, mock_datetime):
        iterations = 0
        cf_unique = []
        it_max = 100
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        count_cf = 1
        score_increase = 1.0
        increase_threshold = 0.0
        activate_tabu = False

        mock_datetime.now.return_value = datetime.datetime(2000, 10, 10, 20, 10, 11)

        result = _greedy_generator_stop_conditions(
            iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
            increase_threshold, activate_tabu)

        self.assertFalse(result)

    def test__generate_greedy_changes_example(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        cf_try = factual.to_numpy()
        tabu_list = []
        changes_cat_bin = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([[-25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        changes_num_down = np.array([[25.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                     [0.0, -5.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])
        avoid_back_original = False

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        self.assertListEqual(changes.tolist(), [[0.,   0.,   0.,   0.,   0.,  -1.,   0.,   0.,   0.,   0.],
                                                [0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.],
                                                [0.,   0.,  -1.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
                                                [0.,   0.,  -1.,   0.,   1.,   0.,   0.,   0.,   0.,   0.],
                                                [0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,  -1.,   0.],
                                                [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -1.,   1.],
                                                [-25., 0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                                                [0.,   5.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                                                [25.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                                                [0.,  -5.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]])

    def test__generate_greedy_changes_avoid_back_original_all_different_from_original(self):
        # In this case, all modifications lead to different values from the original
        factual = pd.Series({'bin1': 1, 'bin2': 0})
        cf_try = np.array([1, 0])
        tabu_list = []
        changes_cat_bin = np.array([[-1.0, 0.0],
                                    [0.0, 1.0]])
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        avoid_back_original = True

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        self.assertListEqual(changes.tolist(), [[-1.,  0.],
                                                [0.,   1.]])

    def test__generate_greedy_changes_avoid_back_original_half_different_from_original(self):
        # In this case, the first modification leads to a value different from original while the second row
        # results in a modification that makes the feature equal to the original (therefore, it's removed)
        factual = pd.Series({'bin1': 1, 'bin2': 1})
        cf_try = np.array([1, 0])
        tabu_list = []
        changes_cat_bin = np.array([[-1.0, 0.0],
                                    [0.0, 1.0]])
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        avoid_back_original = True

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        self.assertListEqual(changes.tolist(), [[-1.,  0.]])

    def test__generate_greedy_changes_avoid_back_original_all_same_as_original(self):
        # In this case, all modifications lead to the same features found in the factual, therefore, no changes
        # should be returned
        factual = pd.Series({'bin1': 0, 'bin2': 1})
        cf_try = np.array([1, 0])
        tabu_list = []
        changes_cat_bin = np.array([[-1.0, 0.0],
                                    [0.0, 1.0]])
        changes_cat_ohe = np.array([])
        changes_num_up = np.array([])
        changes_num_down = np.array([])
        avoid_back_original = True

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        self.assertListEqual(changes.tolist(), [])

    def test__generate_greedy_changes_tabu(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        cf_try = factual.to_numpy()
        tabu_list = [[0], [5], [7, 8, 9]]
        changes_cat_bin = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        changes_cat_ohe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        changes_num_up = np.array([[-25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [-0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        changes_num_down = np.array([[25.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                                     [0.0, -5.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]])
        avoid_back_original = False

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        self.assertListEqual(changes.tolist(), [[0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                                                [0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
                                                [0.,  5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                                [0., -5.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_example(self, mock_greedy_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_greedy_generator_stop_conditions.side_effect = [True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # No CF can be equal to the factual
        self.assertTrue(sum([np.array_equal(cf, factual_np) for cf in cf_try]) == 0)

        # Since we have 10 results with probability above 0.5 we expect 10 results
        self.assertEqual(len(cf_try), 10)

        # Verify if iterations variable was incremented
        self.assertEqual(mock_greedy_generator_stop_conditions.call_args_list[1][1]['iterations'], 2)

    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_only_one_cf_out(self, mock_greedy_generator_stop_conditions):
        # This tests if the function returns only one CF if only one score is above 0.5
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for the first detected modificaiton
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            has_cf = False
            for row_etf in equal_to_factual:
                if row_etf or has_cf:
                    out_result.append(0.0)
                else:
                    has_cf = True
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_greedy_generator_stop_conditions.side_effect = [True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Since we have 1 results with probability above 0.5 we expect 1 results
        self.assertEqual(len(cf_try), 1)

    @patch('cfnow._cf_searchers.deque')
    @patch('cfnow._cf_searchers._generate_greedy_changes')
    @patch('cfnow._cf_searchers._create_factual_changes')
    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_recent_improvements_below_required(
            self, mock_greedy_generator_stop_conditions, mock_create_factual_changes, mock_generate_greedy_changes,
            mock_deque):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(0.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = deque([], maxlen=3)
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_create_factual_changes.side_effect = lambda *args: _create_factual_changes(*args)
        mock_generate_greedy_changes.side_effect = lambda *args: _generate_greedy_changes(*args)
        mock_deque.return_value = deque([0, 0, 0], maxlen=3)
        mock_greedy_generator_stop_conditions.side_effect = [True, True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if momentum was increased in the last step
        self.assertEqual(mock_create_factual_changes.call_args_list[-1][0][3], 1)

        # Verify if Tabu list has one feature in the last step
        self.assertTrue(len(mock_generate_greedy_changes.call_args_list[0][0][2]) > 0)

    @patch('cfnow._cf_searchers.deque')
    @patch('cfnow._cf_searchers._generate_greedy_changes')
    @patch('cfnow._cf_searchers._create_factual_changes')
    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_recent_improvements_below_required_tabu_ohe(
            self, mock_greedy_generator_stop_conditions, mock_create_factual_changes, mock_generate_greedy_changes,
            mock_deque):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(0.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[0, 1, 2], [3, 4, 5]]
        ohe_indexes = [0, 1, 2, 3, 4, 5]
        increase_threshold = 0.0
        tabu_list = deque([], maxlen=3)
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_create_factual_changes.side_effect = lambda *args: _create_factual_changes(*args)
        mock_generate_greedy_changes.side_effect = lambda *args: _generate_greedy_changes(*args)
        mock_deque.return_value = deque([0, 0, 0], maxlen=3)
        mock_greedy_generator_stop_conditions.side_effect = [True, True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if momentum was increased in the last step
        self.assertEqual(mock_create_factual_changes.call_args_list[-1][0][3], 1)

        # Verify if Tabu list has one feature in the last step
        self.assertTrue(len(mock_generate_greedy_changes.call_args_list[0][0][2]) > 0)

        for tabu_list_indexes in list(mock_generate_greedy_changes.call_args_list[0][0][2]):
            # Verify if Tabu list elements are OHE indexes
            self.assertTrue(tabu_list_indexes in ohe_list)

    @patch('cfnow._cf_searchers.deque')
    @patch('cfnow._cf_searchers._generate_greedy_changes')
    @patch('cfnow._cf_searchers._create_factual_changes')
    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_recent_improvements_below_required_reset_momentum(
            self, mock_greedy_generator_stop_conditions, mock_create_factual_changes, mock_generate_greedy_changes,
            mock_deque):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF in the second change (as we have OHE features, it's when we have at least 3
            # modifications)
            if type(x) == pd.DataFrame:
                num_modified_features = factual_np.shape[0] - (x.to_numpy() == factual_np).sum(axis=1)
            else:
                num_modified_features = factual_np.shape[0] - (x == factual_np).sum(axis=1)

            out_result = []
            for row_nmf in num_modified_features:
                if row_nmf < 3:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = deque([], maxlen=3)
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_create_factual_changes.side_effect = lambda *args: _create_factual_changes(*args)
        mock_generate_greedy_changes.side_effect = lambda *args: _generate_greedy_changes(*args)
        mock_deque.return_value = deque([0, 0, 0], maxlen=3)
        mock_greedy_generator_stop_conditions.side_effect = [True, True, True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if momentum was increased in the last step
        self.assertEqual(mock_create_factual_changes.call_args_list[-1][0][3], 0)

        # Verify if Tabu list has one feature in the last step
        self.assertTrue(len(mock_generate_greedy_changes.call_args_list[0][0][2]) > 0)

    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_get_best_cf(self, mock_greedy_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns two counterfactuals
            # [-50, 10, 0, 1, 0, 1, 0, 0, 1, 0] => p = 0.51
            # [-50, 10, 1, 0, 0, 1, 0, 1, 0, 0] => p = 0.81

            cf_tries = x
            if type(x) == pd.DataFrame:
                cf_tries = x.to_numpy()

            out_result = []
            for row in cf_tries:
                if (row != np.array([-50, 10, 0, 1, 0, 1, 0, 0, 1, 0])).sum() == 0:
                    out_result.append(0.51)
                elif (row != np.array([-50, 10, 1, 0, 0, 1, 0, 1, 0, 0])).sum() == 0:
                    out_result.append(0.81)
                else:
                    out_result.append(0.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_greedy_generator_stop_conditions.side_effect = [True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if the first result is the best one
        self.assertListEqual(cf_try[0].tolist(), [-50, 10, 1, 0, 0, 1, 0, 1, 0, 0])

    @patch('cfnow._cf_searchers._generate_greedy_changes')
    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_no_changes(self, mock_greedy_generator_stop_conditions, mock_generate_greedy_changes):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_greedy_generator_stop_conditions.side_effect = [True, False]
        mock_generate_greedy_changes.return_value = np.array([])

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # It must return an empty list
        self.assertTrue(len(cf_try) == 0)

    @patch('cfnow._cf_searchers.logging')
    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_verbose(self, mock_greedy_generator_stop_conditions, mock_logging):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = True

        mock_greedy_generator_stop_conditions.side_effect = [True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        mock_logging.log.assert_called()

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_sequential_update(self, mock_random_generator_stop_conditions):
        finder_strategy = 'sequential'
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if iterations variable was incremented
        self.assertEqual(mock_random_generator_stop_conditions.call_args_list[1][1]['iterations'], 2)

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_random_below_threshold(self, mock_random_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(0.4)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1
        cf_unique = []
        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Verify if iterations variable was incremented
        self.assertEqual(mock_random_generator_stop_conditions.call_args_list[1][1]['iterations'], 2)

    @patch('cfnow._cf_searchers._random_generator_stop_conditions')
    def test__random_generator_no_cf_candidates(self, mock_random_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = None
        tabu_list = None
        size_tabu = 3
        avoid_back_original = None
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1

        cf_unique = [
            [-50.,  10.,   1.,   0.,   0.,   0.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   1.,   0.,   0.,   1.,   1.,   0.,   1.,   0.],
            [-75.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-25.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  15.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,   5.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   0.,   1.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   0.,   0.,   1.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   1.,   0.,   0.],
            [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
            [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   0.,   1.]]
        len_initial_cf_unique = len(cf_unique)

        verbose = False

        mock_random_generator_stop_conditions.side_effect = [True, False]

        cf_try = _random_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # Since we skipped one iteration, we must have CF than the initial value
        self.assertTrue(len(cf_try) > len_initial_cf_unique)

    @patch('cfnow._cf_searchers._greedy_generator_stop_conditions')
    def test__greedy_generator_no_cf_candidate(self, mock_greedy_generator_stop_conditions):
        finder_strategy = None
        cf_data_type = None
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_np = factual.to_numpy()

        mp1c = MagicMock()

        def _mp1c_side_effect_function(x):
            # This function returns a CF for any set different from the factual
            if type(x) == pd.DataFrame:
                equal_to_factual = (x.to_numpy() == factual_np).sum(axis=1) == factual_np.shape[0]
            else:
                equal_to_factual = (x == factual_np).sum(axis=1) == factual_np.shape[0]

            out_result = []
            for row_etf in equal_to_factual:
                if row_etf:
                    out_result.append(0.0)
                else:
                    out_result.append(1.0)

            return np.array(out_result)

        # This function finds a CF for any modification done
        mp1c.side_effect = _mp1c_side_effect_function

        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        it_max = 100
        ft_change_factor = 0.5
        ohe_list = [[2, 3, 4], [7, 8, 9]]
        ohe_indexes = [2, 3, 4, 7, 8, 9]
        increase_threshold = 0.0
        tabu_list = None
        size_tabu = 3
        avoid_back_original = False
        ft_time = datetime.datetime(2000, 10, 10, 10, 10, 10)
        ft_time_limit = 100
        threshold_changes = 1000
        count_cf = 1

        cf_unique = [[-50.,  10.,   1.,   0.,   0.,   0.,   0.,   0.,   1.,   0.],
                     [-50.,  10.,   1.,   0.,   0.,   1.,   1.,   0.,   1.,   0.],
                     [-50.,  10.,   0.,   1.,   0.,   1.,   0.,   0.,   1.,   0.],
                     [-50.,  10.,   0.,   0.,   1.,   1.,   0.,   0.,   1.,   0.],
                     [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   1.,   0.,   0.],
                     [-50.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   0.,   1.],
                     [-75.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
                     [-50.,  15.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
                     [-25.,  10.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.],
                     [-50.,   5.,   1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.]]

        verbose = False

        mock_greedy_generator_stop_conditions.side_effect = [True, False]

        cf_try = _greedy_generator(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            mp1c=mp1c,
            feat_types=feat_types,
            it_max=it_max,
            ft_change_factor=ft_change_factor,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            tabu_list=tabu_list,
            size_tabu=size_tabu,
            avoid_back_original=avoid_back_original,
            ft_time=ft_time,
            ft_time_limit=ft_time_limit,
            threshold_changes=threshold_changes,
            count_cf=count_cf,
            cf_unique=cf_unique,
            verbose=verbose)

        # The iteration must be only one since we have a early stop
        self.assertEqual(mock_greedy_generator_stop_conditions.call_args_list[1][1]['iterations'], 1)
