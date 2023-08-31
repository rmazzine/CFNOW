import unittest
from unittest.mock import patch, MagicMock, call

import cv2
import pandas as pd
import numpy as np

from cfnow.cf_finder import _CFBaseResponse, _CFTabular, _CFImage, _CFText, _define_tabu_size, \
    find_tabular, find_image, find_text


class TestCFBaseResponse(unittest.TestCase):
    def test_create_object(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_vector = factual.to_numpy()
        cf_vectors = [factual.to_numpy()]
        cf_not_optimized_vectors = [factual.to_numpy()]
        obj_scores = [0]
        time_cf = 0
        time_cf_not_optimized = 0
        cf_base = _CFBaseResponse(
            factual=factual, factual_vector=factual_vector, cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors, obj_scores=obj_scores, time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized)
        self.assertEqual(cf_base.total_cf, 1)

    def test_create_object_no_cf(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_vector = factual.to_numpy()
        cf_vectors = []
        cf_not_optimized_vectors = []
        obj_scores = []
        time_cf = 0
        time_cf_not_optimized = 0
        cf_base = _CFBaseResponse(
            factual=factual, factual_vector=factual_vector, cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors, obj_scores=obj_scores, time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized)
        self.assertEqual(cf_base.total_cf, 0)


class TestCFTabular(unittest.TestCase):
    def test_create_object(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_vector = factual.to_numpy()
        cf_vectors = [factual.to_numpy()]
        cf_not_optimized_vectors = [factual.to_numpy()]
        obj_scores = [0]
        time_cf = 0
        time_cf_not_optimized = 0

        cf_tabular = _CFTabular(
            factual=factual, factual_vector=factual_vector, cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors, obj_scores=obj_scores, time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized)

        self.assertListEqual([list(c) for c in cf_tabular.cfs], [list(c) for c in cf_vectors])
        self.assertListEqual(
            [list(c) for c in cf_tabular.cf_not_optimized_vectors], [list(c) for c in cf_not_optimized_vectors])

    def test_create_object_no_cf(self):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        factual_vector = factual.to_numpy()
        cf_vectors = []
        cf_not_optimized_vectors = []
        obj_scores = []
        time_cf = 0
        time_cf_not_optimized = 0

        cf_tabular = _CFTabular(
            factual=factual, factual_vector=factual_vector, cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors, obj_scores=obj_scores, time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized)

        self.assertListEqual(cf_tabular.cfs, [])
        self.assertListEqual(cf_tabular.cf_not_optimized_vectors, [])


class TestCFImage(unittest.TestCase):

    factual = []
    segments = []
    replace_img = []
    for ir in range(240):
        row_img = []
        row_segments = []
        row_replace_image = []
        for ic in range(240):
            row_replace_image.append([0, 0, 0])
            if ir > ic:
                row_img.append([127, 127, 127])
                row_segments.append(1)
            else:
                row_img.append([255, 255, 255])
                row_segments.append(0)
        replace_img.append(row_replace_image)
        factual.append(row_img)
        segments.append(row_segments)
    replace_img = np.array(replace_img).astype('uint8')
    factual = np.array(factual).astype('uint8')
    segments = np.array(segments)

    def test_create_object(self):
        mock_seg_to_img = MagicMock()

        factual_vector = pd.Series({n: 1 for n in range(len(np.unique(self.segments)))})
        cf_vectors = [
            np.array([0]*len(np.unique(self.segments))),
            np.array([1] + [0] * (len(np.unique(self.segments)) - 1)),
        ]
        cf_not_optimized_vectors = [
            np.array([0] * len(np.unique(self.segments))),
            np.array([1] + [0] * (len(np.unique(self.segments)) - 1)),
        ]
        obj_scores = [1, 2]
        time_cf = 0
        time_cf_not_optimized = 0

        cf_image = _CFImage(
            factual=self.factual,
            factual_vector=factual_vector,
            cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors,
            obj_scores=obj_scores,
            time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized,

            _seg_to_img=mock_seg_to_img,
            segments=self.segments,
            replace_img=self.replace_img)

        self.assertListEqual(cf_image.segments.tolist(), self.segments.tolist())

        self.assertListEqual([list(s) for s in cf_image.cfs_segments], [[0, 1], [1]])
        self.assertListEqual([list(s) for s in cf_image.cfs_not_optimized_segments], [[0, 1], [1]])

        green_replace = np.zeros(self.factual.shape)
        green_replace[:, :, 1] = 1

        # Verify if loop correctly assign the right parameters
        self.assertListEqual(mock_seg_to_img.call_args_list[0][0][0][0].tolist(), cf_vectors[0].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[1][0][0][0].tolist(), cf_not_optimized_vectors[0].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[2][0][0][0].tolist(), cf_vectors[1].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[3][0][0][0].tolist(), cf_not_optimized_vectors[1].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[4][0][0][0].tolist(), cf_vectors[0].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[5][0][0][0].tolist(), cf_not_optimized_vectors[0].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[6][0][0][0].tolist(), cf_vectors[1].tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[7][0][0][0].tolist(), cf_not_optimized_vectors[1].tolist())

        self.assertListEqual(mock_seg_to_img.call_args_list[0][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[1][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[2][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[3][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[4][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[5][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[6][0][1].tolist(), self.factual.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[7][0][1].tolist(), self.factual.tolist())

        self.assertListEqual(mock_seg_to_img.call_args_list[0][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[1][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[2][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[3][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[4][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[5][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[6][0][2].tolist(), self.segments.tolist())
        self.assertListEqual(mock_seg_to_img.call_args_list[7][0][2].tolist(), self.segments.tolist())

        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[0][0][3], self.replace_img))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[1][0][3], green_replace))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[2][0][3], self.replace_img))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[3][0][3], green_replace))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[4][0][3], self.replace_img))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[5][0][3], green_replace))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[6][0][3], self.replace_img))
        self.assertTrue(np.array_equal(mock_seg_to_img.call_args_list[7][0][3], green_replace))

    def test_create_object_no_cf(self):
        mock_seg_to_img = MagicMock()

        factual_vector = pd.Series({n: 1 for n in range(len(np.unique(self.segments)))})
        cf_vectors = []
        cf_not_optimized_vectors = []
        obj_scores = []
        time_cf = 0
        time_cf_not_optimized = 0

        cf_image = _CFImage(
            factual=self.factual,
            factual_vector=factual_vector,
            cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors,
            obj_scores=obj_scores,
            time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized,

            _seg_to_img=mock_seg_to_img,
            segments=self.segments,
            replace_img=self.replace_img)

        self.assertListEqual(cf_image.segments.tolist(), self.segments.tolist())

        self.assertListEqual(cf_image.cfs, [])
        self.assertListEqual(cf_image.cfs_image_highlight, [])
        self.assertListEqual(cf_image.cfs_not_optimized, [])
        self.assertListEqual(cf_image.cfs_not_optimized_image_highlight, [])
        self.assertListEqual(cf_image.cfs_segments, [])
        self.assertListEqual(cf_image.cfs_not_optimized_segments, [])

        mock_seg_to_img.assert_not_called()


class TestCFText(unittest.TestCase):

    def test_create_object(self):
        mock_converter = MagicMock()
        text_replace = [['I', ''], ['like', ''], ['music', '']]

        factual = 'I like music'

        factual_vector = np.array([1, 0, 1, 0, 1, 0])
        cf_vectors = [np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])]
        cf_not_optimized_vectors = [np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])]
        obj_scores = [1, 2]
        time_cf = 0
        time_cf_not_optimized = 0

        cf_text = _CFText(
            factual=factual,
            factual_vector=factual_vector,
            cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors,
            obj_scores=obj_scores,
            time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized,

            converter=mock_converter,
            text_replace=text_replace)

        # Verify if loop runs the correct parameters
        self.assertListEqual(mock_converter.call_args_list[0][0][0][0].tolist(), factual_vector.tolist())
        self.assertListEqual(mock_converter.call_args_list[1][0][0][0].tolist(), cf_vectors[0].tolist())
        self.assertListEqual(mock_converter.call_args_list[2][0][0][0].tolist(), cf_not_optimized_vectors[0].tolist())
        self.assertListEqual(mock_converter.call_args_list[3][0][0][0].tolist(), cf_vectors[1].tolist())
        self.assertListEqual(mock_converter.call_args_list[4][0][0][0].tolist(), cf_not_optimized_vectors[1].tolist())
        self.assertListEqual(mock_converter.call_args_list[5][0][0][0].tolist(), cf_vectors[0].tolist())
        self.assertListEqual(mock_converter.call_args_list[6][0][0][0].tolist(), cf_not_optimized_vectors[0].tolist())
        self.assertListEqual(mock_converter.call_args_list[7][0][0][0].tolist(), cf_vectors[1].tolist())
        self.assertListEqual(mock_converter.call_args_list[8][0][0][0].tolist(), cf_not_optimized_vectors[1].tolist())

        self.assertListEqual(cf_text.cfs_replaced_words, [['I', 'like', 'music'], ['like', 'music']])
        self.assertListEqual(cf_text.cfs_not_optimized_replaced_words, [['I', 'like', 'music'], ['like', 'music']])

    def test_create_object_no_cf(self):
        mock_converter = MagicMock()
        text_replace = [['I', ''], ['like', ''], ['music', '']]

        factual = 'I like music'

        factual_vector = np.array([1, 0, 1, 0, 1, 0])
        cf_vectors = []
        cf_not_optimized_vectors = []
        obj_scores = []
        time_cf = 0
        time_cf_not_optimized = 0

        cf_text = _CFText(
            factual=factual,
            factual_vector=factual_vector,
            cf_vectors=cf_vectors,
            cf_not_optimized_vectors=cf_not_optimized_vectors,
            obj_scores=obj_scores,
            time_cf=time_cf,
            time_cf_not_optimized=time_cf_not_optimized,

            converter=mock_converter,
            text_replace=text_replace)

        self.assertListEqual(mock_converter.call_args_list[0][0][0][0].tolist(), factual_vector.tolist())

        self.assertEqual(cf_text.cfs, [])
        self.assertEqual(cf_text.cfs_html_highlight, [])
        self.assertEqual(cf_text.cfs_not_optimized, [])
        self.assertEqual(cf_text.cfs_not_optimized_html_highlight, [])
        self.assertListEqual(cf_text.cfs_replaced_words, [])
        self.assertListEqual(cf_text.cfs_not_optimized_replaced_words, [])


class TestScriptBase(unittest.TestCase):

    # For image CF
    cf_img_default_img = []
    cf_img_default_segments = []
    for ir in range(240):
        row_img = []
        row_segments = []
        for ic in range(240):
            if ir > ic:
                row_img.append([127, 127, 127])
                row_segments.append(1)
            else:
                row_img.append([255, 255, 255])
                row_segments.append(0)
        cf_img_default_img.append(row_img)
        cf_img_default_segments.append(row_segments)
    cf_img_default_img = np.array(cf_img_default_img).astype('uint8')

    cf_img_default_mock_model_predict = MagicMock()
    cf_img_default_count_cf = 1
    cf_img_default_segmentation = 'quickshift'
    cf_img_default_params_segmentation = {}
    cf_img_default_replace_mode = 'blur'
    cf_img_default_img_cf_strategy = 'nonspecific'
    cf_img_default_cf_strategy = 'greedy'
    cf_img_default_increase_threshold = None
    cf_img_default_it_max = 5000
    cf_img_default_limit_seconds = 120
    cf_img_default_ft_change_factor = 0.1
    cf_img_default_ft_it_max = None
    cf_img_default_size_tabu = 0.5
    cf_img_default_ft_threshold_distance = -1.0
    cf_img_default_avoid_back_original = True
    cf_img_default_threshold_changes = 100
    cf_img_default_verbose = False

    cf_img_default_segments = np.array(cf_img_default_segments)

    cf_img_default_factual = pd.Series(np.array([1] * (np.max(cf_img_default_segments) + 1)))
    cf_img_default_replace_img = cv2.GaussianBlur(cf_img_default_img, (31, 31), 0)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_example(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'greedy'
        increase_threshold = 1e-05
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = None
        size_tabu = None
        ft_threshold_distance = 1e-05
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check _check_factual
        mock_check_factual.assert_called_once_with(factual)
        # Check _check_vars
        mock_check_vars.assert_called_once_with(factual, feat_types)
        # Check _check_prob_func
        mock_check_prob_func.assert_called_once_with(factual, model_predict_proba)

        # Check _adjust_model_class
        mock_adjust_model_class.assert_called_once_with(factual, mock_standardize_predictor())

        # Check _get_ohe_params
        mock_get_ohe_params.assert_called_with(factual, False)

        # Check call for cf_finder
        self.assertEqual(len(mock_greedy_generator.call_args[1]), 19)
        self.assertEqual(mock_greedy_generator.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['cf_data_type'], 'tabular')
        self.assertListEqual(mock_greedy_generator.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertEqual(mock_greedy_generator.call_args[1]['mp1c'], mock_mp1c)
        self.assertEqual(mock_greedy_generator.call_args[1]['feat_types'], feat_types)
        self.assertEqual(mock_greedy_generator.call_args[1]['it_max'], it_max)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertListEqual(mock_greedy_generator.call_args[1]['ohe_list'], [[2, 3, 4], [7, 8, 9]])
        self.assertListEqual(mock_greedy_generator.call_args[1]['ohe_indexes'], [2, 3, 4, 7, 8, 9])
        self.assertEqual(mock_greedy_generator.call_args[1]['increase_threshold'], increase_threshold)
        self.assertEqual(mock_greedy_generator.call_args[1]['tabu_list'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 5)
        self.assertEqual(mock_greedy_generator.call_args[1]['avoid_back_original'], avoid_back_original)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time_limit'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['threshold_changes'], 1000)
        self.assertEqual(mock_greedy_generator.call_args[1]['count_cf'], count_cf)
        self.assertEqual(mock_greedy_generator.call_args[1]['cf_unique'], [])
        self.assertEqual(mock_greedy_generator.call_args[1]['verbose'], verbose)

        # Check call for _fine_tuning
        self.assertEqual(len(mock_fine_tuning.call_args[1]), 20)
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_data_type'], 'tabular')
        self.assertListEqual(mock_fine_tuning.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_unique'], mock_greedy_generator())
        self.assertEqual(mock_fine_tuning.call_args[1]['count_cf'], count_cf)
        self.assertEqual(mock_fine_tuning.call_args[1]['mp1c'], mock_mp1c)
        self.assertListEqual(mock_fine_tuning.call_args[1]['ohe_list'], [[2, 3, 4], [7, 8, 9]])
        self.assertListEqual(mock_fine_tuning.call_args[1]['ohe_indexes'], [2, 3, 4, 7, 8, 9])
        self.assertEqual(mock_fine_tuning.call_args[1]['increase_threshold'], increase_threshold)
        self.assertEqual(mock_fine_tuning.call_args[1]['feat_types'], feat_types)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertEqual(mock_fine_tuning.call_args[1]['it_max'], it_max)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 5)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_it_max'], 1000)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], ft_threshold_distance)
        self.assertEqual(mock_fine_tuning.call_args[1]['limit_seconds'], limit_seconds)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_greedy_generator)
        self.assertEqual(mock_fine_tuning.call_args[1]['avoid_back_original'], avoid_back_original)
        self.assertEqual(mock_fine_tuning.call_args[1]['threshold_changes'], 1000)
        self.assertEqual(mock_fine_tuning.call_args[1]['verbose'], verbose)

        # Check call for _CFTabular
        self.assertEqual(len(mock_CFTabular.call_args[1]), 8)
        self.assertListEqual(mock_CFTabular.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertListEqual(mock_CFTabular.call_args[1]['factual_vector'].tolist(), factual.tolist())
        self.assertEqual(mock_CFTabular.call_args[1]['cf_vectors'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFTabular.call_args[1]['cf_not_optimized_vectors'], mock_greedy_generator())
        self.assertEqual(mock_CFTabular.call_args[1]['obj_scores'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFTabular.call_args[1]['time_cf'], mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(
            mock_CFTabular.call_args[1]['time_cf_not_optimized'], mock_datetime.now().__sub__().total_seconds())

        self.assertEqual(response_obj, mock_CFTabular())

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_cf_strategy_random(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'random'
        increase_threshold = 1e-05
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = None
        size_tabu = None
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if _random_generator was called
        mock_random_generator.assert_called_once()

        self.assertEqual(mock_random_generator.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_random_generator)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_cf_strategy_random_sequential(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'random-sequential'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 5000
        size_tabu = 5
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if _random_generator was called
        mock_random_generator.assert_called_once()

        self.assertEqual(mock_random_generator.call_args[1]['finder_strategy'], 'sequential')
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], 'sequential')
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_random_generator)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_it_max'], 5000)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_cf_strategy_error(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'TEST_ERROR'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 5
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = cf_out
        mock_greedy_generator.return_value = cf_out

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        # The function should raise an error if the cf_strategy is not valid
        with self.assertRaises(AttributeError):
            response_obj = find_tabular(
                factual=factual,
                model_predict_proba=model_predict_proba,
                count_cf=count_cf,
                feat_types=feat_types,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                has_ohe=has_ohe,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_no_feat_types(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = None
        cf_strategy = 'greedy'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 5
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        feat_types_all_num = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'num', 'ohe1_1': 'num', 'ohe1_2': 'num',
                              'bin1': 'num', 'bin2': 'num', 'ohe2_0': 'num', 'ohe2_1': 'num', 'ohe2_2': 'num'}

        # Check _check_vars
        mock_check_vars.assert_called_once_with(factual, feat_types_all_num)

        self.assertEqual(mock_greedy_generator.call_args[1]['feat_types'], feat_types_all_num)

        self.assertEqual(mock_fine_tuning.call_args[1]['feat_types'], feat_types_all_num)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_size_tabu_lower(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'greedy'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 4
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert warning is not raised
        mock_warnings.warn.assert_not_called()

        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 4)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 4)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_size_tabu_larger(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'greedy'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 100
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]
        mock_greedy_generator.return_value = [
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([-25, 10, 0, 1, 0, 1, 1, 0, 0, 1])]


        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [1.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert that warning was called
        mock_warnings.warn.assert_called_once()

        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 9)

        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 9)

    @patch('cfnow.cf_finder._CFTabular')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder._adjust_model_class')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._check_prob_func')
    @patch('cfnow.cf_finder._check_vars')
    @patch('cfnow.cf_finder._check_factual')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_tabular_no_cf_found(
            self, mock_random_generator, mock_greedy_generator, mock_warnings, mock_check_factual, mock_check_vars,
            mock_check_prob_func, mock_standardize_predictor, mock_adjust_model_class, mock_get_ohe_params,
            mock_datetime, mock_logging, mock_fine_tuning, mock_CFTabular):
        factual = pd.Series({'num1': -50, 'num2': 10, 'ohe1_0': 1, 'ohe1_1': 0, 'ohe1_2': 0, 'bin1': 1, 'bin2': 0,
                             'ohe2_0': 0, 'ohe2_1': 1, 'ohe2_2': 0})
        model_predict_proba = MagicMock()
        count_cf = 1
        feat_types = {'num1': 'num', 'num2': 'num', 'ohe1_0': 'cat', 'ohe1_1': 'cat', 'ohe1_2': 'cat',
                      'bin1': 'cat', 'bin2': 'cat', 'ohe2_0': 'cat', 'ohe2_1': 'cat', 'ohe2_2': 'cat'}
        cf_strategy = 'greedy'
        increase_threshold = 0
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 5
        ft_threshold_distance = 0.01
        has_ohe = False
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        mock_get_ohe_params.return_value = [[2, 3, 4], [7, 8, 9]], [2, 3, 4, 7, 8, 9]

        cf_out = np.array([-25, 10, 0, 1, 0, 1, 1, 0, 1, 0])
        mock_random_generator.return_value = []
        mock_greedy_generator.return_value = []

        mock_mp1c = MagicMock()
        mock_mp1c.return_value = [0.0]
        mock_adjust_model_class.return_value = mock_mp1c

        response_obj = find_tabular(
            factual=factual,
            model_predict_proba=model_predict_proba,
            count_cf=count_cf,
            feat_types=feat_types,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            has_ohe=has_ohe,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert logging was called
        mock_logging.log.assert_called_once()

        # Check response object
        self.assertEqual(len(mock_CFTabular.call_args[1]), 7)
        self.assertListEqual(mock_CFTabular.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertListEqual(mock_CFTabular.call_args[1]['factual_vector'].tolist(), factual.tolist())
        self.assertEqual(mock_CFTabular.call_args[1]['cf_vectors'], [])
        self.assertEqual(mock_CFTabular.call_args[1]['cf_not_optimized_vectors'], [])
        self.assertEqual(mock_CFTabular.call_args[1]['obj_scores'], [])
        self.assertEqual(mock_CFTabular.call_args[1]['time_cf'], mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(
            mock_CFTabular.call_args[1]['time_cf_not_optimized'], mock_datetime.now().__sub__().total_seconds())

        self.assertEqual(response_obj, mock_CFTabular())

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_example(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check gen_quickshift was called
        mock_gen_quickshift.assert_called_once_with(img, params_segmentation)

        # Check _adjust_image_model was called
        self.assertEqual(len(mock_adjust_image_model.call_args[0]), 4)
        self.assertListEqual(mock_adjust_image_model.call_args[0][0].tolist(), img.tolist())
        self.assertEqual(mock_adjust_image_model.call_args[0][1], mock_model_predict)
        self.assertListEqual(mock_adjust_image_model.call_args[0][2].tolist(), segments.tolist())
        self.assertListEqual(mock_adjust_image_model.call_args[0][3].tolist(), replace_img.tolist())

        # Check _adjust_multiclass_nonspecific was called
        self.assertEqual(len(mock_adjust_multiclass_nonspecific.call_args[0]), 2)
        self.assertListEqual(mock_adjust_multiclass_nonspecific.call_args[0][0].tolist(),
                             self.cf_img_default_factual.tolist())
        self.assertEqual(mock_adjust_multiclass_nonspecific.call_args[0][1], mock_adjust_image_model())

        # Check _adjust_multiclass_second_best was not called
        self.assertFalse(mock_adjust_multiclass_second_best.called)

        # Check the call made by _greedy_generator
        self.assertEqual(len(mock_greedy_generator.call_args[1]), 19)
        self.assertEqual(mock_greedy_generator.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['cf_data_type'], 'image')
        self.assertEqual(mock_greedy_generator.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertEqual(mock_greedy_generator.call_args[1]['mp1c'], mock_adjust_multiclass_nonspecific())
        self.assertEqual(mock_greedy_generator.call_args[1]['feat_types'], {0: 'cat', 1: 'cat'})
        self.assertEqual(mock_greedy_generator.call_args[1]['it_max'], 5000)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertEqual(mock_greedy_generator.call_args[1]['ohe_list'], [])
        self.assertEqual(mock_greedy_generator.call_args[1]['ohe_indexes'], [])
        self.assertEqual(mock_greedy_generator.call_args[1]['increase_threshold'], -1.0)
        self.assertEqual(mock_greedy_generator.call_args[1]['tabu_list'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['avoid_back_original'], True)
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time_limit'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['count_cf'], count_cf)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time_limit'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['cf_unique'], [])
        self.assertEqual(mock_greedy_generator.call_args[1]['threshold_changes'], 100)

        self.assertEqual(mock_greedy_generator.call_args[1]['verbose'], verbose)

        # Check _fine_tuning call
        self.assertEqual(len(mock_fine_tuning.call_args[1]), 20)
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_data_type'], 'image')
        self.assertEqual(mock_fine_tuning.call_args[1]['factual'].tolist(), factual.tolist())
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_unique'], mock_greedy_generator())
        self.assertEqual(mock_fine_tuning.call_args[1]['count_cf'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['mp1c'], mock_adjust_multiclass_nonspecific())
        self.assertEqual(mock_fine_tuning.call_args[1]['ohe_list'], [])
        self.assertEqual(mock_fine_tuning.call_args[1]['ohe_indexes'], [])
        self.assertEqual(mock_fine_tuning.call_args[1]['increase_threshold'], -1.0)
        self.assertEqual(mock_fine_tuning.call_args[1]['feat_types'], {0: 'cat', 1: 'cat'})
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertEqual(mock_fine_tuning.call_args[1]['it_max'], 5000)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_it_max'], 1000)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], 0.01)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], -1)
        self.assertEqual(mock_fine_tuning.call_args[1]['time_start'], mock_datetime.now())

        self.assertEqual(mock_fine_tuning.call_args[1]['limit_seconds'], 120)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_greedy_generator)
        self.assertEqual(mock_fine_tuning.call_args[1]['avoid_back_original'], True)
        self.assertEqual(mock_fine_tuning.call_args[1]['threshold_changes'], 100)
        self.assertEqual(mock_fine_tuning.call_args[1]['verbose'], False)

        # Check call for _CFImage
        self.assertEqual(len(mock_CFImage.call_args[1]), 10)
        self.assertListEqual(mock_CFImage.call_args[1]['factual'].tolist(), img.tolist())
        self.assertListEqual(mock_CFImage.call_args[1]['factual_vector'].tolist(), factual.tolist())
        self.assertEqual(mock_CFImage.call_args[1]['cf_vectors'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFImage.call_args[1]['cf_not_optimized_vectors'], mock_greedy_generator())
        self.assertEqual(mock_CFImage.call_args[1]['obj_scores'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFImage.call_args[1]['time_cf'], mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFImage.call_args[1]['time_cf_not_optimized'],
                         mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFImage.call_args[1]['_seg_to_img'], mock_seg_to_img)
        self.assertListEqual(mock_CFImage.call_args[1]['segments'].tolist(), segments.tolist())
        self.assertEqual(mock_CFImage.call_args[1]['replace_img'].tolist(), replace_img.tolist())

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_random_strategy(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy

        cf_strategy = 'random'

        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_random_generator.return_value = [np.array([0, 1]), np.array([1, 0])]


        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,

            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if _random_generator was called
        mock_random_generator.assert_called()

        self.assertEqual(mock_random_generator.call_args[1]['it_max'], 5000)
        self.assertEqual(mock_random_generator.call_args[1]['increase_threshold'], 1e-05)
        self.assertEqual(mock_random_generator.call_args[1]['it_max'], 5000)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], -1.0)
        self.assertEqual(mock_fine_tuning.call_args[1]['threshold_changes'], 100)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_error_cf_strategy(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):
        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy

        cf_strategy = 'TEST_ERROR'

        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        # Check if exception is raised
        with self.assertRaises(AttributeError):
            response_obj = find_image(
                img=img,
                model_predict=mock_model_predict,
                count_cf=count_cf,
                segmentation=segmentation,
                params_segmentation=params_segmentation,
                replace_mode=replace_mode,
                img_cf_strategy=img_cf_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_segmentation_error(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf

        segmentation = 'TEST_ERROR'

        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        # Check if exception is raised
        with self.assertRaises(AttributeError):
            response_obj = find_image(
                img=img,
                model_predict=mock_model_predict,
                count_cf=count_cf,
                segmentation=segmentation,
                params_segmentation=params_segmentation,
                replace_mode=replace_mode,
                img_cf_strategy=img_cf_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_replace_mean(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):
        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation

        replace_mode = 'mean'

        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments

        replace_img = np.zeros(img.shape)
        replace_img[:, :, 0], replace_img[:, :, 1], replace_img[:, :, 2] = img.mean(axis=(0, 1))

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertListEqual(mock_adjust_image_model.call_args[0][3].tolist(), replace_img.tolist())

        self.assertEqual(mock_CFImage.call_args[1]['replace_img'].tolist(), replace_img.tolist())

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder.np')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_replace_random(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_np, mock_CFImage):
        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation

        replace_mode = 'random'

        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        mock_np.side_effect = np

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock_np.random.random.assert_called_once_with(img.shape)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_replace_inpaint(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):
        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation

        replace_mode = 'inpaint'

        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments

        replace_img = np.zeros(img.shape)
        for j in np.unique(segments):
            image_absolute = (img * 255).astype('uint8')
            mask = np.full([image_absolute.shape[0], image_absolute.shape[1]], 0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            replace_img[segments == j] = image_segment_inpainted[segments == j] / 255.0

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertListEqual(mock_adjust_image_model.call_args[0][3].tolist(), replace_img.tolist())

        self.assertEqual(mock_CFImage.call_args[1]['replace_img'].tolist(), replace_img.tolist())

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder.np')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_replace_error(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_np, mock_CFImage):
        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation

        replace_mode = 'TEST_ERROR'

        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments

        segments = np.array(segments)
        mock_gen_quickshift.return_value = segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        mock_np.side_effect = np

        with self.assertRaises(AttributeError):
            find_image(
                img=img,
                model_predict=mock_model_predict,
                count_cf=count_cf,
                segmentation=segmentation,
                params_segmentation=params_segmentation,
                replace_mode=replace_mode,
                img_cf_strategy=img_cf_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_size_tabu_larger(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max

        size_tabu = 1

        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert warning is not raised
        mock_warnings.warn.assert_not_called()

        # Verify if Tabu is the length of the segments minus 1 (1)
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 1)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_size_tabu_larger(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max

        size_tabu = 1000

        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert warning is raised
        mock_warnings.warn.assert_called_once()

        # Verify if Tabu is the length of the segments minus 1 (1)
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 1)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_cf_img_strategy_second_best(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode

        img_cf_strategy = 'second_best'

        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_mimns_sb = MagicMock()
        mock_mimns_sb.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns_sb
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertEqual(mock_greedy_generator.call_args[1]['mp1c'], mock_adjust_multiclass_second_best())
        self.assertEqual(mock_fine_tuning.call_args[1]['mp1c'], mock_adjust_multiclass_second_best())

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_cf_img_strategy_error(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode

        img_cf_strategy = 'TEST_ERROR'

        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_mimns_sb = MagicMock()
        mock_mimns_sb.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns_sb
        mock_greedy_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        with self.assertRaises(AttributeError):
            find_image(
                img=img,
                model_predict=mock_model_predict,
                count_cf=count_cf,
                segmentation=segmentation,
                params_segmentation=params_segmentation,
                replace_mode=replace_mode,
                img_cf_strategy=img_cf_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_return_no_cf(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy
        cf_strategy = self.cf_img_default_cf_strategy
        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        avoid_back_original = self.cf_img_default_avoid_back_original
        threshold_changes = self.cf_img_default_threshold_changes
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [0.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_greedy_generator.return_value = []

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check call for _CFImage
        self.assertEqual(len(mock_CFImage.call_args[1]), 10)
        self.assertListEqual(mock_CFImage.call_args[1]['factual'].tolist(), img.tolist())
        self.assertListEqual(mock_CFImage.call_args[1]['factual_vector'].tolist(), factual.tolist())
        self.assertEqual(mock_CFImage.call_args[1]['cf_vectors'], [])
        self.assertEqual(mock_CFImage.call_args[1]['cf_not_optimized_vectors'], [])
        self.assertEqual(mock_CFImage.call_args[1]['obj_scores'], [])
        self.assertEqual(mock_CFImage.call_args[1]['time_cf'], mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFImage.call_args[1]['time_cf_not_optimized'],
                         mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFImage.call_args[1]['_seg_to_img'], mock_seg_to_img)
        self.assertListEqual(mock_CFImage.call_args[1]['segments'].tolist(), segments.tolist())
        self.assertEqual(mock_CFImage.call_args[1]['replace_img'].tolist(), replace_img.tolist())

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder._define_tabu_size')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_example(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_define_tabu_size, mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]
        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1.0
        it_max = 5000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = None
        size_tabu = None
        ft_threshold_distance = None
        avoid_back_original = False
        threshold_changes = 100
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_greedy_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if _text_to_token_vector was called
        mock_text_to_token_vector.assert_called_once_with(text_input)

        # Check if _convert_change_vectors_func was called with the right parameters
        mock_convert_change_vectors_func.assert_called_once_with(
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])

        # Check if _adjust_textual_classifier was called with the right parameters
        mock_adjust_textual_classifier.assert_called_once_with(
            mock_textual_classifier,
            mock_convert_change_vectors_func(),
            [0.0])

        # Check if _standardize_predictor was called with the right parameters
        self.assertListEqual(mock_standardize_predictor.call_args[0][0].tolist(), factual_df.iloc[0].tolist())
        self.assertEqual(mock_standardize_predictor.call_args[0][1], mock_adjust_textual_classifier())

        # Check calls from mts
        self.assertListEqual(mock_mts.call_args_list[0][0][0].to_numpy()[0].tolist(), factual_df.iloc[0].tolist())

        # Check if _get_ohe_params was called with the right parameters
        self.assertListEqual(mock_get_ohe_params.call_args[0][0].tolist(), factual_df.iloc[0].tolist())
        self.assertEqual(mock_get_ohe_params.call_args[0][1], True)

        # Check if define_tabu_size was called with the right parameters
        self.assertEqual(len(mock_define_tabu_size.call_args[0]), 2)
        self.assertEqual(mock_define_tabu_size.call_args[0][0], 5)
        self.assertEqual(mock_define_tabu_size.call_args[0][1].tolist(), factual_df.iloc[0].to_numpy().tolist())

        # Check if cf_finder was called with the right parameters
        self.assertEqual(len(mock_greedy_generator.call_args[1]), 19)
        self.assertEqual(mock_greedy_generator.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['cf_data_type'], 'text')
        self.assertEqual(mock_greedy_generator.call_args[1]['factual'].tolist(), factual_df.iloc[0].tolist())
        self.assertEqual(mock_greedy_generator.call_args[1]['mp1c'], mock_mts)
        self.assertEqual(mock_greedy_generator.call_args[1]['feat_types'],
                         {'0_0': 'cat', '0_1': 'cat', '1_0': 'cat', '1_1': 'cat', '2_0': 'cat', '2_1': 'cat'})
        self.assertEqual(mock_greedy_generator.call_args[1]['it_max'], it_max)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertEqual(mock_greedy_generator.call_args[1]['ohe_list'], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(mock_greedy_generator.call_args[1]['ohe_indexes'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(mock_greedy_generator.call_args[1]['increase_threshold'], -1)
        self.assertEqual(mock_greedy_generator.call_args[1]['tabu_list'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['avoid_back_original'], avoid_back_original)
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], mock_define_tabu_size())
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['ft_time_limit'], None)
        self.assertEqual(mock_greedy_generator.call_args[1]['threshold_changes'], 100)
        self.assertEqual(mock_greedy_generator.call_args[1]['count_cf'], count_cf)
        self.assertListEqual(mock_greedy_generator.call_args[1]['cf_unique'], [])

        self.assertEqual(mock_greedy_generator.call_args[1]['verbose'], verbose)

        # Check _fine_tuning called with the right parameters
        self.assertEqual(len(mock_fine_tuning.call_args[1]), 20)
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], None)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_data_type'], 'text')
        self.assertEqual(mock_fine_tuning.call_args[1]['factual'].tolist(), factual_df.iloc[0].tolist())
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_unique'], mock_greedy_generator())
        self.assertEqual(mock_fine_tuning.call_args[1]['count_cf'], count_cf)
        self.assertEqual(mock_fine_tuning.call_args[1]['mp1c'], mock_mts)
        self.assertEqual(mock_fine_tuning.call_args[1]['ohe_list'], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(mock_fine_tuning.call_args[1]['ohe_indexes'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(mock_fine_tuning.call_args[1]['increase_threshold'], -1)
        self.assertEqual(mock_fine_tuning.call_args[1]['feat_types'],
                         {'0_0': 'cat', '0_1': 'cat', '1_0': 'cat', '1_1': 'cat', '2_0': 'cat', '2_1': 'cat'})
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_change_factor'], ft_change_factor)
        self.assertEqual(mock_fine_tuning.call_args[1]['it_max'], it_max)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], mock_define_tabu_size())

        self.assertEqual(mock_fine_tuning.call_args[1]['ft_it_max'], 2000)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], -1.0)
        self.assertEqual(mock_fine_tuning.call_args[1]['limit_seconds'], 120)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_greedy_generator)
        self.assertEqual(mock_fine_tuning.call_args[1]['avoid_back_original'], avoid_back_original)
        self.assertEqual(mock_fine_tuning.call_args[1]['threshold_changes'], 100)
        self.assertEqual(mock_fine_tuning.call_args[1]['verbose'], False)

        # Check call for _CFText
        self.assertEqual(len(mock_CFText.call_args[1]), 9)
        self.assertEqual(mock_CFText.call_args[1]['factual'], text_input)
        self.assertListEqual(mock_CFText.call_args[1]['factual_vector'].tolist(), factual_df.iloc[0].tolist())
        self.assertEqual(mock_CFText.call_args[1]['cf_vectors'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFText.call_args[1]['cf_not_optimized_vectors'], mock_greedy_generator())
        self.assertEqual(mock_CFText.call_args[1]['obj_scores'], mock_fine_tuning().__getitem__())
        self.assertEqual(mock_CFText.call_args[1]['time_cf'], mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFText.call_args[1]['time_cf_not_optimized'],
                         mock_datetime.now().__sub__().total_seconds())

        self.assertEqual(mock_CFText.call_args[1]['converter'], mock_convert_change_vectors_func())
        self.assertEqual(mock_CFText.call_args[1]['text_replace'], [['I', ''], ['like', ''], ['music', '']])

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_cf_finder_random(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'random'
        increase_threshold = -1.0
        it_max = 5000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = None
        size_tabu = None
        ft_threshold_distance = None
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])

        mock_random_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock_random_generator.assert_called_once()

        # Check call for _fine_tuning
        self.assertEqual(mock_random_generator.call_args[1]['increase_threshold'], -1.0)
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_random_generator)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_it_max'], 100)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 1)
        self.assertEqual(mock_fine_tuning.call_args[1]['ft_threshold_distance'], 1e-05)

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_cf_finder_error(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'TEST_ERROR'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        # The function should raise an error
        with self.assertRaises(AttributeError):
            response_obj = find_text(
                text_input=text_input,
                textual_classifier=mock_textual_classifier,
                count_cf=count_cf,
                word_replace_strategy=word_replace_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_word_strategy_antonyms(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'antonyms'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'1_0': 1, '1_1': 0, '1_2': 0, '1_3': 0}])
        mock_text_to_change_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [[], ['like', 'unlike', 'unalike', 'dislike'], []])

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1, 2, 3]], [0, 1, 2, 3])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        mock_text_to_change_vector.assert_called_once_with(text_input)

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_word_strategy_error(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'TEST_ERROR'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        # The error is raised when the word_replace_strategy is not valid
        with self.assertRaises(AttributeError):
            response_obj = find_text(
                text_input=text_input,
                textual_classifier=mock_textual_classifier,
                count_cf=count_cf,
                word_replace_strategy=word_replace_strategy,
                cf_strategy=cf_strategy,
                increase_threshold=increase_threshold,
                it_max=it_max,
                limit_seconds=limit_seconds,
                ft_change_factor=ft_change_factor,
                ft_it_max=ft_it_max,
                size_tabu=size_tabu,
                ft_threshold_distance=ft_threshold_distance,
                avoid_back_original=avoid_back_original,
                threshold_changes=threshold_changes,
                verbose=verbose)


    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_adjusted_classification_different_same_class(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.1]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_greedy_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if logging was called
        mock_logging.log.assert_called_once()

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_adjusted_classification_different_different_class(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.6]
            else:
                return [0.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        mock_greedy_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if logging was called
        mock_logging.log.assert_called_once()

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_adjusted_classification_different_changed_to_cf(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [0.6]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_greedy_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.6]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.6]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if logging was called
        mock_logging.log.assert_called_once()

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_tabu_lower(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000

        size_tabu = 2

        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_greedy_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert warning is not raised
        mock_warnings.warn.assert_not_called()

        # Check if Tabu list size is correct
        self.assertEqual(mock_greedy_generator.call_args[1]['size_tabu'], 2)
        self.assertEqual(mock_fine_tuning.call_args[1]['size_tabu'], 2)

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_no_cf_found(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [0.0]

        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'greedy'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_greedy_generator.return_value = []

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [0.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Assert logging was called
        mock_logging.log.assert_called_once()

        # Check call for _CFText
        self.assertEqual(len(mock_CFText.call_args[1]), 9)
        self.assertEqual(mock_CFText.call_args[1]['factual'], text_input)
        self.assertListEqual(mock_CFText.call_args[1]['factual_vector'].tolist(), factual_df.iloc[0].tolist())
        self.assertListEqual(mock_CFText.call_args[1]['cf_vectors'], [])
        self.assertListEqual(mock_CFText.call_args[1]['cf_not_optimized_vectors'], [])
        self.assertListEqual(mock_CFText.call_args[1]['obj_scores'], [])
        self.assertEqual(mock_CFText.call_args[1]['time_cf'],
                         mock_datetime.now().__sub__().total_seconds())
        self.assertEqual(mock_CFText.call_args[1]['time_cf_not_optimized'],
                         mock_datetime.now().__sub__().total_seconds())

        self.assertEqual(mock_CFText.call_args[1]['converter'], mock_convert_change_vectors_func())
        self.assertEqual(mock_CFText.call_args[1]['text_replace'], [['I', ''], ['like', ''], ['music', '']])

    def test_define_tabu_size_float_example(self):
        size_tabu = 0.5
        factual_vector = pd.Series({'0': 1, '1': 0, '2': 1, '3': 0, '4': 1, '5': 0})
        self.assertEqual(_define_tabu_size(size_tabu, factual_vector), 3)

    def test_define_tabu_size_int_example(self):
        size_tabu = 4
        factual_vector = pd.Series({'0': 1, '1': 0, '2': 1, '3': 0, '4': 1, '5': 0})
        self.assertEqual(_define_tabu_size(size_tabu, factual_vector), 4)

    def test_define_tabu_size_float_larger_than_1(self):
        size_tabu = 1.5
        factual_vector = pd.Series({'0': 1, '1': 0, '2': 1, '3': 0, '4': 1, '5': 0})
        with self.assertRaises(AttributeError):
            _define_tabu_size(size_tabu, factual_vector)

    def test_define_tabu_size_float_lower_than_0(self):
        size_tabu = -0.5
        factual_vector = pd.Series({'0': 1, '1': 0, '2': 1, '3': 0, '4': 1, '5': 0})
        with self.assertRaises(AttributeError):
            _define_tabu_size(size_tabu, factual_vector)

    @patch('cfnow.cf_finder.warnings')
    def test_define_tabu_size_int_larger_than_factual_vector_size(self, mock_warnings):
        size_tabu = 7
        factual_vector = pd.Series({'0': 1, '1': 0, '2': 1, '3': 0, '4': 1, '5': 0})

        out_size_tabu = _define_tabu_size(size_tabu, factual_vector)

        # Assert warnings was called
        mock_warnings.warn.assert_called_once()

        # Assert the out_size_tabu is the length of the factual vector minus 1
        self.assertEqual(out_size_tabu, factual_vector.size - 1)

    @patch('cfnow.cf_finder._CFImage')
    @patch('cfnow.cf_finder._seg_to_img')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._adjust_multiclass_second_best')
    @patch('cfnow.cf_finder._adjust_multiclass_nonspecific')
    @patch('cfnow.cf_finder._adjust_image_model')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder.gen_quickshift')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test_find_image_example(
            self, mock_random_generator, mock_greedy_generator, mock_gen_quickshift, mock_warnings,
            mock_adjust_image_model, mock_adjust_multiclass_nonspecific, mock_adjust_multiclass_second_best,
            mock_datetime, mock_logging, mock_fine_tuning, mock_seg_to_img, mock_CFImage):

        img = self.cf_img_default_img
        mock_model_predict = self.cf_img_default_mock_model_predict
        count_cf = self.cf_img_default_count_cf
        segmentation = self.cf_img_default_segmentation
        params_segmentation = self.cf_img_default_params_segmentation
        replace_mode = self.cf_img_default_replace_mode
        img_cf_strategy = self.cf_img_default_img_cf_strategy

        cf_strategy = 'random-sequential'

        increase_threshold = self.cf_img_default_increase_threshold
        it_max = self.cf_img_default_it_max
        limit_seconds = self.cf_img_default_limit_seconds
        ft_change_factor = self.cf_img_default_ft_change_factor
        ft_it_max = self.cf_img_default_ft_it_max
        size_tabu = self.cf_img_default_size_tabu
        ft_threshold_distance = self.cf_img_default_ft_threshold_distance
        threshold_changes = self.cf_img_default_threshold_changes
        avoid_back_original = self.cf_img_default_avoid_back_original
        verbose = self.cf_img_default_verbose

        factual = self.cf_img_default_factual
        segments = self.cf_img_default_segments
        replace_img = self.cf_img_default_replace_img

        mock_gen_quickshift.return_value = self.cf_img_default_segments

        mock_mimns = MagicMock()
        mock_mimns.return_value = [1.0]

        mock_adjust_multiclass_nonspecific.return_value = mock_mimns
        mock_adjust_multiclass_second_best.return_value = mock_mimns
        mock_random_generator.return_value = [np.array([0, 1]), np.array([1, 0])]

        response_obj = find_image(
            img=img,
            model_predict=mock_model_predict,
            count_cf=count_cf,
            segmentation=segmentation,
            params_segmentation=params_segmentation,
            replace_mode=replace_mode,
            img_cf_strategy=img_cf_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        self.assertEqual(mock_random_generator.call_args[1]['finder_strategy'], 'sequential')

        # Check _fine_tuning call
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], 'sequential')
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_random_generator)

    @patch('cfnow.cf_finder._CFText')
    @patch('cfnow.cf_finder._define_tabu_size')
    @patch('cfnow.cf_finder.warnings')
    @patch('cfnow.cf_finder._fine_tuning')
    @patch('cfnow.cf_finder._get_ohe_params')
    @patch('cfnow.cf_finder.logging')
    @patch('cfnow.cf_finder._standardize_predictor')
    @patch('cfnow.cf_finder._adjust_textual_classifier')
    @patch('cfnow.cf_finder._convert_change_vectors_func')
    @patch('cfnow.cf_finder._text_to_change_vector')
    @patch('cfnow.cf_finder._text_to_token_vector')
    @patch('cfnow.cf_finder.datetime')
    @patch('cfnow.cf_finder._greedy_generator')
    @patch('cfnow.cf_finder._random_generator')
    def test__find_text_random_sequential(
            self, mock_random_generator, mock_greedy_generator, mock_datetime, mock_text_to_token_vector,
            mock_text_to_change_vector, mock_convert_change_vectors_func, mock_adjust_textual_classifier,
            mock_standardize_predictor, mock_logging, mock_get_ohe_params, mock_fine_tuning, mock_warnings,
            mock_define_tabu_size, mock_CFText):
        text_input = 'I like music'

        def _textual_classifier_predict(array_txt_input):
            if array_txt_input[0] == 'I like music':
                return [0.0]
            else:
                return [1.0]
        mock_textual_classifier = MagicMock()
        mock_textual_classifier.side_effect = lambda x: _textual_classifier_predict(x)

        count_cf = 1
        word_replace_strategy = 'remove'
        cf_strategy = 'random-sequential'
        increase_threshold = -1
        it_max = 1000
        limit_seconds = 120
        ft_change_factor = 0.1
        ft_it_max = 1000
        size_tabu = 0.5
        ft_threshold_distance = 0.01
        avoid_back_original = False
        threshold_changes = 1000
        verbose = False

        factual_df = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        mock_text_to_token_vector.return_value = (
            ['I', 'like', 'music'],
            factual_df,
            [['I', ''], ['like', ''], ['music', '']])
        mock_random_generator.return_value = [
            np.array([0, 1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        ]

        def _mock_mts(factual):
            if type(factual) == pd.DataFrame:
                if np.array_equal(factual.to_numpy(), factual_df.to_numpy()):
                    return [0.0]
            if type(factual) == np.ndarray:
                if np.array_equal(factual, factual_df.to_numpy()):
                    return [0.0]

            return [1.0]

        mock_mts = MagicMock()
        mock_mts.side_effect = _mock_mts

        mock_standardize_predictor.return_value = mock_mts

        mock_get_ohe_params.return_value = ([[0, 1], [2, 3], [4, 5]], [0, 1, 2, 3, 4, 5])

        response_obj = find_text(
            text_input=text_input,
            textual_classifier=mock_textual_classifier,
            count_cf=count_cf,
            word_replace_strategy=word_replace_strategy,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes,
            verbose=verbose)

        # Check if cf_finder was called with the right parameters
        self.assertEqual(mock_random_generator.call_args[1]['finder_strategy'], 'sequential')

        # Check _fine_tuning called with the right parameters
        self.assertEqual(mock_fine_tuning.call_args[1]['finder_strategy'], 'sequential')
        self.assertEqual(mock_fine_tuning.call_args[1]['cf_finder'], mock_random_generator)
