"""
This module has the input checkers for the CF generator.
"""
import pandas as pd


def _check_factual(factual):
    # Factual must be a pandas Series
    try:
        assert type(factual) == pd.Series
    except AssertionError:
        raise TypeError(f'Factual must be a Pandas Series. However it is {type(factual)}.')


def _check_vars(factual, feat_types):
    # The number of feat_types must be the same as the number of factual features
    missing_var = []
    extra_var = []
    try:
        missing_var = list(set(factual.index) - set(feat_types.keys()))
        extra_var = list(set(feat_types.keys()) - set(factual.index))
        assert len(missing_var) == 0 and len(extra_var) == 0
    except AssertionError:
        if len(missing_var) > 0 and len(extra_var) > 0:
            raise AssertionError(f"\nThe features:\n {','.join(missing_var)}\n"
                                 f"must have their type defined in feat_types.\
                                 \n\nAnd the features:\n {','.join(extra_var)}\n"
                                 f"are not defined in the factual point")
        elif len(missing_var) > 0:
            raise AssertionError(
                f"The features:\n {','.join(missing_var)}\nmust have their type defined in feat_types.")
        elif len(extra_var) > 0:
            raise AssertionError(f"The features:\n {','.join(extra_var)}\nare not defined in the factual point.")


def _check_prob_func(factual, model_predict_proba):
    # Test model function and get the classification of factual
    try:
        model_predict_proba(factual.to_frame().T)
    except Exception:
        raise Exception('Error when using the model_predict_proba function.')
