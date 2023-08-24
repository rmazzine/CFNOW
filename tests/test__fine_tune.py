import unittest
import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from cfnow._fine_tune import _create_mod_change, _calculate_change_factor, _generate_change_vectors, \
    _stop_optimization_conditions, _fine_tuning


class TestScriptBase(unittest.TestCase):
    def test__create_mod_change_numerical(self):
        changes_back_factual = [1]
        change_idx = 0
        change_original_idx = 0
        ft_change_factor = 10
        def _feat_idx_to_type(x): return 'num'

        mod_change_result = _create_mod_change(changes_back_factual, change_idx, change_original_idx,
                                               ft_change_factor, _feat_idx_to_type)

        self.assertEqual(mod_change_result, 10)

    def test__create_mod_change_categorical(self):
        changes_back_factual = [1]
        change_idx = 0
        change_original_idx = 0
        ft_change_factor = 10
        def _feat_idx_to_type(x): return 'cat'

        mod_change_result = _create_mod_change(changes_back_factual, change_idx, change_original_idx,
                                               ft_change_factor, _feat_idx_to_type)

        self.assertEqual(mod_change_result, 1)

    def test__calculate_change_factor(self):
        c_cf = np.array([1, 1, 1, 1])
        changes_back_factual = np.array([[0.1, 0., 0., 0.], [0., 0.1, 0., 0.], [0., 0., 0.1, 0.], [0., 0., 0., 0.1]])
        feat_distances = np.array([1, 0.8, 0.4, 0.2])
        changes_back_original_idxs = [0, 1, 2, 3]
        mp1c = MagicMock()
        mp1c.return_value = np.array([0.9, 0.9, 0.9, 0.9])
        c_cf_c = 1.0

        change_factor_feat = _calculate_change_factor(c_cf, changes_back_factual, feat_distances,
                                                      changes_back_original_idxs,  mp1c, c_cf_c)

        self.assertListEqual([0.1, 0.125, 0.25, 0.5], list([round(n, 3) for n in change_factor_feat]))
        self.assertIsInstance(change_factor_feat, np.ndarray)

    def test__calculate_change_factor_negative(self):
        c_cf = np.array([1, 1, 1, 1])
        changes_back_factual = np.array([[0.1, 0., 0., 0.], [0., 0.1, 0., 0.], [0., 0., 0.1, 0.], [0., 0., 0., 0.1]])
        feat_distances = np.array([1, 0.8, 0.4, 0.2])
        changes_back_original_idxs = [0, 1, 2, 3]
        mp1c = MagicMock()
        mp1c.return_value = np.array([0.7, 0.7, 0.5, 0.5])
        c_cf_c = 0.6

        change_factor_feat = _calculate_change_factor(c_cf, changes_back_factual, feat_distances,
                                                      changes_back_original_idxs,  mp1c, c_cf_c)

        self.assertListEqual([-0.1, -0.08, 0.25, 0.5], list([round(n, 3) for n in change_factor_feat]))
        self.assertIsInstance(change_factor_feat, np.ndarray)

    def test__calculate_change_factor_positive_zero_distance(self):
        # This should not happen as the objective function should not return zero values
        c_cf = np.array([1, 1, 1, 1])
        changes_back_factual = np.array([[0.1, 0., 0., 0.], [0., 0.1, 0., 0.], [0., 0., 0.1, 0.], [0., 0., 0., 0.1]])
        feat_distances = np.array([1, 0.8, 0, 0.2])
        changes_back_original_idxs = [0, 1, 2, 3]
        mp1c = MagicMock()
        mp1c.return_value = np.array([0.7, 0.7, 0.5, 0.5])
        c_cf_c = 0.6

        change_factor_feat = _calculate_change_factor(c_cf, changes_back_factual, feat_distances,
                                                      changes_back_original_idxs,  mp1c, c_cf_c)

        self.assertListEqual([-0.1, -0.08, 0, 0.5], list([round(n, 3) for n in change_factor_feat]))
        self.assertIsInstance(change_factor_feat, np.ndarray)

    def test__generate_change_vectors_all_num_no_tabu(self):
        factual = pd.Series({0: 1, 1: 5, 2: 10, 3: 50})
        factual_np = factual.to_numpy()
        c_cf = np.array([1, 5, 5, 25])
        def _feat_idx_to_type(x): return 'num'
        tabu_list = []
        ohe_indexes = []
        ohe_list = []
        ft_threshold_distance = 0
        unique_cf = [np.array([1, 5, 5, 25])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[0., 0., 5., 0.], [0., 0., 0., 25.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [2, 3])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [0, 0, 5, 25])

    def test__generate_change_vectors_all_bin_no_tabu(self):
        factual = pd.Series({0: 1, 1: 0, 2: 0, 3: 1})
        factual_np = factual.to_numpy()
        c_cf = np.array([0, 0, 1, 1])
        def _feat_idx_to_type(x): return 'cat'
        tabu_list = []
        ohe_indexes = []
        ohe_list = []
        ft_threshold_distance = 0
        unique_cf = [np.array([0, 0, 1, 1])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[1., 0., 0., 0.], [0., 0., -1., 0.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [0, 2])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [1, 0, 1, 0])

    def test__generate_change_vectors_all_ohe_no_tabu(self):
        factual = pd.Series({0: 1, 1: 0, 2: 1, 3: 0})
        factual_np = factual.to_numpy()
        c_cf = np.array([1, 0, 0, 1])
        def _feat_idx_to_type(x): return 'cat'
        tabu_list = []
        ohe_indexes = [0, 1, 2, 3]
        ohe_list = [[0, 1], [2, 3]]
        ft_threshold_distance = 0
        unique_cf = [np.array([1, 0, 0, 1])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[0., 0., 1., -1.], [0., 0., 1., -1.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [2, 3])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [0, 0, 1, 1])

    def test__generate_change_vectors_num_bin_ohe_no_tabu(self):
        # idx num: 0*, 1
        # idx ohe: [2, 3, 4], [5, 6, 7]*
        # idx bin: 8*, 9
        # Asterisks means the features which were modified
        factual = pd.Series({0: 50, 1: 100, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 1})
        factual_np = factual.to_numpy()
        c_cf = np.array([25, 100, 1, 0, 0, 0, 0, 1, 1, 1])
        def _feat_idx_to_type(x): return 'cat' if x not in [0, 1] else 'num'
        tabu_list = []
        ohe_indexes = [2, 3, 4, 5, 6, 7]
        ohe_list = [[2, 3, 4], [5, 6, 7]]
        ft_threshold_distance = 0
        unique_cf = [np.array([25, 100, 1, 0, 0, 0, 0, 1, 1, 1])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual],
                             [
                                 [25., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., -1., 0.]
                             ])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [0, 5, 7, 8])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [25., 0., 0., 0., 0., 1., 0., 1., 1., 0.])

    def test__generate_change_vectors_non_unique_changes(self):
        # idx num: 0*, 1
        # idx ohe: [2, 3, 4], [5, 6, 7]*
        # idx bin: 8*, 9
        # Asterisks means the features which were modified
        factual = pd.Series({0: 50, 1: 100, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 1})
        factual_np = factual.to_numpy()
        c_cf = np.array([25, 100, 1, 0, 0, 0, 0, 1, 1, 1])
        def _feat_idx_to_type(x): return 'cat' if x not in [0, 1] else 'num'
        tabu_list = []
        ohe_indexes = [2, 3, 4, 5, 6, 7]
        ohe_list = [[2, 3, 4], [5, 6, 7]]
        ft_threshold_distance = 0
        unique_cf = [
            np.array([25, 100, 1, 0, 0, 0, 0, 1, 1, 1]),
            np.array([50, 100, 1, 0, 0, 0, 0, 1, 1, 1]),
            np.array([25, 100, 1, 0, 0, 0, 0, 1, 0, 1]),
        ]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual],
                             [
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                             ])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [5, 7])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [25., 0., 0., 0., 0., 1., 0., 1., 1., 0.])

    def test__generate_change_vectors_all_num_with_tabu(self):
        factual = pd.Series({0: 1, 1: 5, 2: 10, 3: 50})
        factual_np = factual.to_numpy()
        c_cf = np.array([1, 5, 5, 25])
        def _feat_idx_to_type(x): return 'num'
        tabu_list = [[2]]
        ohe_indexes = []
        ohe_list = []
        ft_threshold_distance = 0
        unique_cf = [np.array([1, 5, 5, 25])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[0., 0., 0., 25.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [3])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [0, 0, 5, 25])

    def test__generate_change_vectors_all_bin_with_tabu(self):
        factual = pd.Series({0: 1, 1: 0, 2: 0, 3: 1})
        factual_np = factual.to_numpy()
        c_cf = np.array([0, 0, 1, 1])
        def _feat_idx_to_type(x): return 'cat'
        tabu_list = [[2]]
        ohe_indexes = []
        ohe_list = []
        ft_threshold_distance = 0
        unique_cf = [np.array([0, 0, 1, 1])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[1., 0., 0., 0.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [0])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [1, 0, 1, 0])

    def test__generate_change_vectors_all_ohe_with_tabu(self):
        factual = pd.Series({0: 1, 1: 0, 2: 1, 3: 0})
        factual_np = factual.to_numpy()
        c_cf = np.array([0, 1, 0, 1])
        def _feat_idx_to_type(x): return 'cat'
        tabu_list = [[0, 1]]
        ohe_indexes = [0, 1, 2, 3]
        ohe_list = [[0, 1], [2, 3]]
        ft_threshold_distance = 0
        unique_cf = [np.array([0, 1, 0, 1])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[0., 0., 1., -1.], [0., 0., 1., -1.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [2, 3])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [1, 1, 1, 1])

    def test__generate_change_vectors_num_bin_ohe_with_tabu(self):
        # idx num: 0*tabu, 1*
        # idx ohe: [2, 3, 4]*tabu, [5, 6, 7]*
        # idx bin: 8*tabu, 9*
        # Asterisks means the features which were modified
        factual = pd.Series({0: 50, 1: 100, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 1})
        factual_np = factual.to_numpy()
        c_cf = np.array([25, 50, 0, 1, 0, 0, 0, 1, 1, 0])
        def _feat_idx_to_type(x): return 'cat' if x not in [0, 1] else 'num'
        tabu_list = [[0], [2, 3, 4], [8]]
        ohe_indexes = [2, 3, 4, 5, 6, 7]
        ohe_list = [[2, 3, 4], [5, 6, 7]]
        ft_threshold_distance = 0
        unique_cf = [np.array([25, 50, 0, 1, 0, 0, 0, 1, 1, 0])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual],
                             [
                                 [0., 50., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
                             ])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [1, 5, 7, 9])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [25., 50., 1., 1., 0., 1., 0., 1., 1., 1.])

    def test__generate_change_vectors_modification_threshold(self):
        factual = pd.Series({0: 1, 1: 5, 2: 10, 3: 50})
        factual_np = factual.to_numpy()
        c_cf = np.array([1, 5, 5, 49.9])

        def _feat_idx_to_type(x): return 'num'

        tabu_list = []
        ohe_indexes = []
        ohe_list = []
        ft_threshold_distance = 0.2
        unique_cf = [np.array([1, 5, 5, 49.9])]

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual=factual, factual_np=factual_np, c_cf=c_cf, _feat_idx_to_type=_feat_idx_to_type,
            tabu_list=tabu_list, ohe_indexes=ohe_indexes, ohe_list=ohe_list,
            ft_threshold_distance=ft_threshold_distance, unique_cf=unique_cf)

        # What are the changes that must be made to go back to the original values
        self.assertListEqual([list(c) for c in changes_back_factual], [[0., 0., 5., 0.]])

        # Which features are different from original
        self.assertListEqual(changes_back_original_idxs, [2])

        # Array with the distances of each feature
        self.assertListEqual(list(feat_distances), [0, 0, 5, 0])

    @patch('cfnow._fine_tune.datetime')
    @patch('cfnow._fine_tune._obj_manhattan')
    def test__stop_optimization_conditions_dont_stop(self, mock__obj_manhattan, mock_datetime):
        mock_datetime.now.return_value = datetime.datetime.fromtimestamp(1627279000.0)
        mock__obj_manhattan.return_value = 0
        factual_np = np.array([0, 0, 1, 0])
        c_cf = np.array([1, 0, 0, 1])
        limit_seconds = 100
        time_start = datetime.datetime.fromtimestamp(1627279000.0)
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ohe_list = [[2, 3]]

        result_stop = _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list)

        self.assertFalse(result_stop)

    @patch('cfnow._fine_tune.datetime')
    @patch('cfnow._fine_tune._obj_manhattan')
    def test__stop_optimization_conditions_bin(self, mock__obj_manhattan, mock_datetime):
        mock_datetime.now.return_value = datetime.datetime.fromtimestamp(1627279000.0)
        mock__obj_manhattan.return_value = 1
        factual_np = np.array([0])
        c_cf = np.array([1])
        limit_seconds = 100
        time_start = datetime.datetime.fromtimestamp(1627279000.0)
        feat_types = {0: 'cat'}
        ohe_list = []

        result_stop = _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list)

        self.assertTrue(result_stop)

    @patch('cfnow._fine_tune.datetime')
    @patch('cfnow._fine_tune._obj_manhattan')
    def test__stop_optimization_conditions_ohe(self, mock__obj_manhattan, mock_datetime):
        mock_datetime.now.return_value = datetime.datetime.fromtimestamp(1627279000.0)
        mock__obj_manhattan.return_value = 2
        factual_np = np.array([0, 1])
        c_cf = np.array([1, 0])
        limit_seconds = 100
        time_start = datetime.datetime.fromtimestamp(1627279000.0)
        feat_types = {0: 'cat', 1: 'cat'}
        ohe_list = [[0, 1]]

        result_stop = _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list)

        self.assertTrue(result_stop)

    @patch('cfnow._fine_tune.datetime')
    @patch('cfnow._fine_tune._obj_manhattan')
    def test__stop_optimization_conditions_timeout(self, mock__obj_manhattan, mock_datetime):
        mock_datetime.now.return_value = datetime.datetime.fromtimestamp(1627279200.0)
        mock__obj_manhattan.return_value = 0
        factual_np = np.array([0, 0, 1, 0])
        c_cf = np.array([1, 0, 0, 1])
        limit_seconds = 100
        time_start = datetime.datetime.fromtimestamp(1627279000.0)
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ohe_list = [[2, 3]]

        result_stop = _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list)

        self.assertTrue(result_stop)

    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_one_iteration(self, mock__stop_optimization_conditions):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})

        def _mp1c(x): return np.array([1.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        def _cf_finder(x): return np.array([[0, 2, 0, 1]])

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__stop_optimization_conditions.return_value = False

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=_cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertIsInstance(result_finetune[0], np.ndarray)
        self.assertIsInstance(result_finetune[1], np.ndarray)

    @patch('cfnow._fine_tune.deque')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_one_back_factual_not_ohe(self, mock__stop_optimization_conditions, mock_deque):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})

        def _mp1c(x): return np.array([0.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        cf_finder = MagicMock()

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__stop_optimization_conditions.return_value = False

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertListEqual(list(result_finetune[0][0]), list(cf_unique[0]))

        cf_finder.assert_called_once()
        mock_deque().append.assert_called_once_with([0])

    @patch('cfnow._fine_tune.deque')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_one_back_factual_ohe(self, mock__stop_optimization_conditions, mock_deque):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 1, 2: 1, 3: 0})

        def _mp1c(x): return np.array([0.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([1, 1, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        cf_finder = MagicMock()

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__stop_optimization_conditions.return_value = False

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertListEqual(list(result_finetune[0][0]), list(cf_unique[0]))

        cf_finder.assert_called_once()
        mock_deque().append.assert_called_once_with([2, 3])

    @patch('cfnow._fine_tune._create_mod_change')
    @patch('cfnow._fine_tune._calculate_change_factor')
    @patch('cfnow._fine_tune._generate_change_vectors')
    @patch('cfnow._fine_tune._obj_manhattan')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_found_best(
            self, mock__stop_optimization_conditions, mock__obj_manhattan, mock__generate_change_vectors,
            mock__calculate_change_factor, mock__create_mod_change):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})

        def _mp1c(x): return np.array([1.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        def _cf_finder(x): return np.array([[0, 2, 0, 1]])

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__obj_manhattan.return_value = 0
        mock__stop_optimization_conditions.return_value = True

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=_cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock__generate_change_vectors.assert_not_called()
        mock__calculate_change_factor.assert_not_called()
        mock__create_mod_change.assert_not_called()

        self.assertListEqual(list(result_finetune[0][0]), list(cf_unique[0]))
        self.assertEqual(result_finetune[1][0], 0)

    @patch('cfnow._fine_tune.logging')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_verbose(self, mock__stop_optimization_conditions, mock_logging):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})
        cf_out = np.array([0, 2, 0, 1])

        def _mp1c(x): return np.array([1.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        def _cf_finder(x): return np.array([[0, 2, 0, 1]])

        avoid_back_original = False
        threshold_changes = 1000
        verbose = True

        mock__stop_optimization_conditions.return_value = False

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=_cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock_logging.log.assert_called()

    @patch('cfnow._fine_tune._calculate_change_factor')
    @patch('cfnow._fine_tune._create_mod_change')
    @patch('cfnow._fine_tune.warnings')
    @patch('cfnow._fine_tune._generate_change_vectors')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_no_changes(self, mock__stop_optimization_conditions,
                                                    mock__generate_change_vectors, mock_warnings,
                                                    mock__create_mod_change, mock__calculate_change_factor):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})
        cf_out = np.array([0, 2, 0, 1])

        def _mp1c(x): return np.array([1.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        def _cf_finder(x): return np.array([[0, 2, 0, 1]])

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__stop_optimization_conditions.return_value = False

        mock__generate_change_vectors.return_value = (np.array([]), [], np.array([0, 0, 0, 0]))

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=_cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock_warnings.warn.assert_called()

        mock__calculate_change_factor.assert_not_called()
        mock__create_mod_change.assert_not_called()

    @patch('cfnow._fine_tune._calculate_change_factor')
    @patch('cfnow._fine_tune._generate_change_vectors')
    @patch('cfnow._fine_tune._obj_manhattan')
    @patch('cfnow._fine_tune._stop_optimization_conditions')
    def test__fine_tuning_found_new_cf(
            self, mock__stop_optimization_conditions, mock__obj_manhattan, mock__generate_change_vectors,
            mock__calculate_change_factor):
        finder_strategy = None
        cf_data_type = 'tabular'
        factual = pd.Series({0: 1, 1: 2, 2: 1, 3: 0})

        def _mp1c(x): return np.array([0.0] * len(x))

        ohe_list = [[2, 3]]
        ohe_indexes = [2, 3]
        increase_threshold = 0
        feat_types = {0: 'cat', 1: 'num', 2: 'cat', 3: 'cat'}
        ft_change_factor = 0
        it_max = 100
        size_tabu = 5
        ft_it_max = 1
        ft_threshold_distance = 0
        cf_unique = [np.array([0, 2, 0, 1])]
        count_cf = 1
        limit_seconds = 100

        # Additionally to the already found cf, we will find a new one ([0, 2, 0, 1])
        def _cf_finder(**kwargs): return np.array([[0, 2, 0, 1], [0, 4, 0, 1]])

        mock__calculate_change_factor.return_value = [0, 1, 2]

        mock__generate_change_vectors.return_value = (
            np.array([[1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, -1]]), [0, 2, 3], np.array([1, 0, 1, -1]))

        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock__obj_manhattan.return_value = 0
        mock__stop_optimization_conditions.return_value = False

        result_finetune = _fine_tuning(
            finder_strategy=finder_strategy,
            cf_data_type=cf_data_type,
            factual=factual,
            cf_unique=cf_unique,
            count_cf=count_cf,
            mp1c=_mp1c,
            ohe_list=ohe_list,
            ohe_indexes=ohe_indexes,
            increase_threshold=increase_threshold,
            feat_types=feat_types,
            ft_change_factor=ft_change_factor,
            it_max=it_max,
            size_tabu=size_tabu,
            ft_it_max=ft_it_max,
            ft_threshold_distance=ft_threshold_distance,
            limit_seconds=limit_seconds,
            cf_finder=_cf_finder,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertListEqual([list(c) for c in result_finetune[0]], [[0, 2, 0, 1], [0, 4, 0, 1]])
        self.assertEqual(list(result_finetune[1]), [0, 0])


if __name__ == '__main__':
    unittest.main()
