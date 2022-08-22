import unittest
from unittest.mock import call, patch, MagicMock

import numpy as np
import pandas as pd

from cfnow._model_standardizer import _standardize_predictor, _adjust_model_class, _adjust_image_model, \
    _convert_to_numpy, _adjust_multiclass_nonspecific, _adjust_multiclass_second_best, \
    _adjust_textual_classifier


class TestScriptBase(unittest.TestCase):

    factual = pd.Series([0])

    factual_np_single = np.array([0])
    factual_np_multiple = np.array([[0], [0], [0]])

    # Simple image, segments and replace_img to be used in tests
    img = []
    segments = []
    replace_img = []
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
        replace_img.append(row_replace_image)
        img.append(row_img)
        segments.append(row_segments)
    replace_img = np.array(replace_img).astype('uint8')
    img = np.array(img).astype('uint8')
    segments = np.array(segments)

    def test__standardize_predictor_single_1_multiple_30(self):
        # The predictor returns:
        # * Single prediction = Num
        # * Multiple prediction = [Num, Num, Num]
        # Objective is always: single = [Num] , multiple [Num, Num, Num]
        def _model_predict_proba(x): return 0 if len(x) == 1 else [0, 0, 0]

        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

    def test__standardize_predictor_single_1_multiple_31(self):
        # The predictor returns:
        # * Single prediction = Num
        # * Multiple prediction = [[Num], [Num], [Num]]
        # Objective is always: single = [Num] , multiple [Num, Num, Num]
        def _model_predict_proba(x): return 0 if len(x) == 1 else [[0], [0], [0]]

        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

    def test__standardize_predictor_single_10_multiple_30(self):
        # The predictor returns:
        # * Single prediction = [Num]
        # * Multiple prediction = [Num, Num, Num]
        # Objective is always: single = [Num] , multiple [Num, Num, Num]
        def _model_predict_proba(x): return [0] if len(x) == 1 else [0, 0, 0]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

    def test__standardize_predictor_single_10_multiple_31(self):
        # The predictor returns:
        # * Single prediction = [Num]
        # * Multiple prediction = [[Num], [Num], [Num]]
        # Objective is always: single = [Num] , multiple [Num, Num, Num]
        # The result must be a numpy array
        def _model_predict_proba(x): return [0] if len(x) == 1 else [[0], [0], [0]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

    @patch('cfnow._model_standardizer._adjust_multiclass_nonspecific')
    def test__standardize_predictor_single_10_multiple_31_multiclass(self, mock_adjust_multiclass_nonspecific):
        # The predictor returns:
        # * Single prediction = [Num1, Num2, Num3]
        # * Multiple prediction = [[Num1, Num2, Num3], [Num1, Num2, Num3], [Num1, Num2, Num3]]
        # For a multiclass prediction, we use the nonspecific strategy. Then, the factual is the
        # highest class (3rd element)
        # The result must be a numpy array
        # Results must be, respectively, <0.5, ==0.5 and >0.5

        mock_adjust_multiclass_nonspecific.side_effect = _adjust_multiclass_nonspecific

        _model_predict_proba = MagicMock()
        _model_predict_proba.side_effect = lambda x:  [0, 1, 2] if len(x) == 1 else [[0, 1, 2], [2, 1, 2], [3, 1, 2]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

        # Check prediction values
        self.assertTrue(multiple_prediction[0] < 0.5)
        self.assertTrue(multiple_prediction[1] == 0.5)
        self.assertTrue(multiple_prediction[2] > 0.5)

        # The first argument of the mock_adjust_multiclass_nonspecific function must be a numpy array
        self.assertIsInstance(mock_adjust_multiclass_nonspecific.call_args[0][0], np.ndarray)

    @patch('cfnow._model_standardizer._adjust_multiclass_nonspecific')
    def test__standardize_predictor_single_10_multiple_31_two_classes(self, mock_adjust_multiclass_nonspecific):
        # The predictor returns:
        # * Single prediction = [Num1, Num2]
        # * Multiple prediction = [[Num1, Num2], [Num1, Num2], [Num1, Num2]]
        # We use a nonspecific strategy that must work for 2 input
        # Results must be, respectively, <0.5, ==0.5 and >0.5

        mock_adjust_multiclass_nonspecific.side_effect = _adjust_multiclass_nonspecific

        def _model_predict_proba(x): return [0, 1] if len(x) == 1 else [[0, 1], [1, 1], [1, 0]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

        # Check prediction values
        self.assertTrue(multiple_prediction[0] < 0.5)
        self.assertTrue(multiple_prediction[1] == 0.5)
        self.assertTrue(multiple_prediction[2] > 0.5)

        # The first argument of the mock_adjust_multiclass_nonspecific function must be a numpy array
        self.assertIsInstance(mock_adjust_multiclass_nonspecific.call_args[0][0], np.ndarray)

    def test__standardize_predictor_single_11_multiple_31(self):
        # The predictor returns:
        # * Single prediction = [[Num]]
        # * Multiple prediction = [[Num], [Num], [Num]]
        # Objective is always: single = [Num] , multiple [Num, Num, Num]
        def _model_predict_proba(x): return [[0]] if len(x) == 1 else [[0], [0], [0]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

    def test__standardize_predictor_single_11_multiple_31_multiclass(self):
        # The predictor returns:
        # * Single prediction = [[Num1, Num2, Num3]]
        # * Multiple prediction = [[Num1, Num2, Num3], [Num1, Num2, Num3], [Num1, Num2, Num3]]
        # For a multiclass prediction, we use the nonspecific strategy. Then, the factual is the
        # highest class (3rd element)
        # The result must be a numpy array
        # Results must be, respectively, <0.5, ==0.5 and >0.5
        def _model_predict_proba(x): return [[0, 1]] if len(x) == 1 else [[0, 1], [1, 1], [1, 0]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

        # Check prediction values
        self.assertTrue(multiple_prediction[0] < 0.5)
        self.assertTrue(multiple_prediction[1] == 0.5)
        self.assertTrue(multiple_prediction[2] > 0.5)

    def test__standardize_predictor_single_11_multiple_31_two_classes(self):
        # The predictor returns:
        # * Single prediction = [[Num1, Num2]]
        # * Multiple prediction = [[Num1, Num2], [Num1, Num2], [Num1, Num2]]
        # We use a nonspecific strategy that must work for 2 input
        # Results must be, respectively, <0.5, ==0.5 and >0.5
        def _model_predict_proba(x): return [[0, 1, 2]] if len(x) == 1 else [[0, 1, 2], [2, 1, 2], [3, 1, 2]]
        mp1 = _standardize_predictor(self.factual, _model_predict_proba)

        # Single prediction test
        single_prediction = mp1(self.factual_np_single)
        self.assertTupleEqual(single_prediction.shape, (1,))
        self.assertIsInstance(single_prediction, np.ndarray)

        # Multiple prediction test
        multiple_prediction = mp1(self.factual_np_multiple)
        self.assertCountEqual(multiple_prediction.shape, (3,))
        self.assertIsInstance(multiple_prediction, np.ndarray)

        # Check prediction values
        self.assertTrue(multiple_prediction[0] < 0.5)
        self.assertTrue(multiple_prediction[1] == 0.5)
        self.assertTrue(multiple_prediction[2] > 0.5)

    def test__adjust_model_class_model_returns_1_must_return_0(self):
        def _mp1(x): return np.array([1])
        self.assertEqual(_adjust_model_class(self.factual, _mp1)(self.factual)[0], 0)

    def test__adjust_model_class_model_returns_0_must_return_0(self):
        def _mp1(x): return np.array([0])
        self.assertEqual(_adjust_model_class(self.factual, _mp1)(self.factual)[0], 0)

    @patch('cfnow._model_standardizer._seg_to_img')
    def test__adjust_image_model(self, mock__seg_to_img):
        mock__seg_to_img.return_value = [0]

        model_predict = MagicMock()

        _seg_to_img_calls = [call(self.img), call(self.img), call(self.segments), call(self.replace_img)]

        _adjusted_img_model = _adjust_image_model(self.img, model_predict, self.segments, self.replace_img)

        _adjusted_img_model(self.segments)

        mock__seg_to_img.assert_called_once_with(self.segments, self.img, self.segments, self.replace_img)

        model_predict.assert_called_once_with(np.array([0]))

        self.assertIsInstance(model_predict.call_args[0][0], np.ndarray)

    def test__convert_to_numpy_from_pd_dataframe(self):
        a = pd.DataFrame([[0], [0]])
        a_np = _convert_to_numpy(a)

        self.assertIsInstance(a_np, np.ndarray)

    def test__convert_to_numpy_from_np(self):
        a = np.array([[0], [0]])
        a_np = _convert_to_numpy(a)

        self.assertIsInstance(a_np, np.ndarray)

    def test__adjust_multiclass_nonspecific_input_shape_test(self):
        # Test if the input shape conversion works right as data can have shape (n, 1) or (n,)
        _mic = MagicMock()

        _mic.side_effect = lambda x: np.array([[1., 0., 0., 0., 0.]])

        adjusted_predictor_n_0 = _adjust_multiclass_nonspecific(np.array([0, 0, 0, 0]), _mic)

        adjusted_predictor_n_1 = _adjust_multiclass_nonspecific(np.array([[0, 0, 0, 0]]), _mic)

        self.assertListEqual(_mic.call_args_list[0][0][0].tolist(), _mic.call_args_list[1][0][0].tolist())


    def test__adjust_multiclass_nonspecific_factual_larger(self):
        # In this case, the first index is the factual and any other index can be a counterfactual
        def _mic(x): return np.array([[1., 0., 0., 0., 0.]]) if x[0] == 0 else np.array([[100., 0., 0., 0., 0.]])

        adjusted_predictor = _adjust_multiclass_nonspecific(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertEqual(prediction, 0.0)

    def test__adjust_multiclass_nonspecific_factual_same(self):
        # In this case, the first index is the factual and any other index can be a counterfactual
        def _mic(x): return np.array([[1., 0., 0., 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 0., 0., 0.]])

        adjusted_predictor = _adjust_multiclass_nonspecific(np.array([0]), _mic)

        prediction = adjusted_predictor(np.array([1]))[0]

        self.assertTrue(prediction < 0.5)

    def test__adjust_multiclass_nonspecific_counterfactual_larger(self):
        # In this case, the first index is the factual and any other index can be a counterfactual
        def _mic(x): return np.array([[1., 0., 0., 0., 0.]]) if x[0] == 0 else np.array([[0., 0., 100., 0., 0.]])

        adjusted_predictor = _adjust_multiclass_nonspecific(np.array([0]), _mic)

        prediction = adjusted_predictor(np.array([1]))[0]

        self.assertTrue(prediction > 0.5)

    def test__adjust_multiclass_nonspecific_factual_equal_to_counterfactual(self):
        # In this case, the first index is the factual and any other index can be a counterfactual
        def _mic(x): return np.array([[1., 0., 0., 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 1., 0., 0.]])

        adjusted_predictor = _adjust_multiclass_nonspecific(np.array([0]), _mic)

        prediction = adjusted_predictor(np.array([1]))[0]

        self.assertTrue(prediction == 0.5)


    def test__adjust_multiclass_second_best_input_shape_test(self):
        # Test if the input shape conversion works right as data can have shape (n, 1) or (n,)
        _mic = MagicMock()

        _mic.side_effect = lambda x: np.array([[1., 0., 0., 0., 0.]])

        adjusted_predictor_n_0 = _adjust_multiclass_second_best(np.array([0, 0, 0, 0]), _mic)

        adjusted_predictor_n_1 = _adjust_multiclass_second_best(np.array([[0, 0, 0, 0]]), _mic)

        self.assertListEqual(_mic.call_args_list[0][0][0].tolist(), _mic.call_args_list[1][0][0].tolist())


    def test__adjust_multiclass_second_best_factual_larger(self):
        # In this case, first index is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third index equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third index) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[100., 0., 0.5, 0., 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertEqual(prediction, 0.0)

    def test__adjust_multiclass_second_best_factual_same(self):
        # In this case, first index is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third index equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third index) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 0.5, 0., 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction < 0.5)

    def test__adjust_multiclass_second_best_other_equal(self):
        # In this case, first index is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third index equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third index) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 0.5, 0.5, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction < 0.5)

    def test__adjust_multiclass_second_best_other_larger(self):
        # In this case, first element is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third element equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third element) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 0.5, 100.0, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction < 0.5)

    def test__adjust_multiclass_second_best_counterfactual_larger(self):
        # In this case, first element is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third element equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third element) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 100.0, 1.0, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction > 0.5)

    def test__adjust_multiclass_second_best_factual_counterfactual_other_equal(self):
        # In this case, first element is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third element equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third element) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 1.0, 1.0, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction == 0.5)

    def test__adjust_multiclass_second_best_factual_counterfactual_equal(self):
        # In this case, first element is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third element equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third element) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[1., 0., 1.0, 0.0, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction == 0.5)

    def test__adjust_multiclass_second_best_counterfactual_other_equal(self):
        # In this case, first element is the factual, but the counterfactual is the second best in the factual score
        # (in our examples is the third element equal to 0.5), then, a counterfactual is only valid when this feature
        # (the second best in factual score, therefore, the third element) is the largest number
        def _mic(x): return np.array([[1., 0., 0.5, 0., 0.]]) if x[0] == 0 else np.array([[0., 0., 1.0, 1.0, 0.]])

        adjusted_predictor = _adjust_multiclass_second_best(np.array([0]), _mic)

        prediction = round(adjusted_predictor(np.array([1]))[0], 2)

        self.assertTrue(prediction == 0.5)

    def test__adjust_textual_classifier_factual_0(self):
        textual_classifier = MagicMock()
        converter = MagicMock()
        converter.return_value = True
        original_text_classification = 0
        textual_classifier.return_value = 0
        array_texts = [0]

        textual_predictor = _adjust_textual_classifier(textual_classifier, converter, original_text_classification)

        prediction = textual_predictor(array_texts)

        textual_classifier.assert_called_once_with(True)
        converter.assert_called_once_with([0])

        self.assertEqual(prediction, 0)

    def test__adjust_textual_classifier_factual_1(self):
        textual_classifier = MagicMock()
        converter = MagicMock()
        converter.return_value = True
        original_text_classification = 1
        textual_classifier.return_value = 1
        array_texts = [0]

        textual_predictor = _adjust_textual_classifier(textual_classifier, converter, original_text_classification)

        prediction = textual_predictor(array_texts)

        textual_classifier.assert_called_once_with(True)
        converter.assert_called_once_with([0])

        self.assertEqual(prediction, 0)


if __name__ == '__main__':
    unittest.main()
