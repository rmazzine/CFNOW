import unittest
from unittest.mock import MagicMock

import pandas as pd

from cfnow._checkers import _check_factual, _check_vars, _check_prob_func


class TestScriptBase(unittest.TestCase):

    def test__check_factual_not_pd_series(self):

        factual = [0, 1, 2, 3, 4]

        with self.assertRaises(TypeError):
            _check_factual(factual)

    def test__check_factual_is_pd_series(self):

        factual = pd.Series([0, 1, 2, 3, 4])

        _check_factual(factual)

    def test__check_vars_missing_factual_and_feat_types(self):

        factual = pd.Series({'a': 1, 'c': 1})
        feat_types = {'a': 'num', 'b': 'num'}

        with self.assertRaises(AssertionError):
            _check_vars(factual, feat_types)

    def test__check_vars_missing_factual(self):

        factual = pd.Series({'a': 1})
        feat_types = {'a': 'num', 'b': 'num'}

        with self.assertRaises(AssertionError):
            _check_vars(factual, feat_types)

    def test__check_vars_missing_feat_types(self):

        factual = pd.Series({'a': 1, 'c': 1})
        feat_types = {'a': 'num'}

        with self.assertRaises(AssertionError):
            _check_vars(factual, feat_types)

    def test__check_vars_correct(self):

        factual = pd.Series({'a': 1, 'b': 1, 'c': 1})
        feat_types = {'a': 'num', 'b': 'num', 'c': 'num'}

        _check_vars(factual, feat_types)

    def test__check_prob_func_error(self):
        factual = pd.Series({'a': 1, 'b': 1})
        model_predict_proba = MagicMock()
        model_predict_proba.side_effect = Exception()

        with self.assertRaises(Exception):
            _check_prob_func(factual, model_predict_proba)

    def test__check_prob_func_correct(self):
        factual = pd.Series({'a': 1, 'b': 1})
        model_predict_proba = MagicMock()
