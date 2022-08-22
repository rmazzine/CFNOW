"""
This module has the functions which take a model prediction function (which returns probability) and return
a standardized function that can be used by the CF generator.
"""
import copy

import numpy as np
import pandas as pd

from ._data_standardizer import _seg_to_img


def _standardize_predictor(factual, model_predict_proba):
    # Convert the output of prediction function to something that can be treated
    # mp1 always return the 1 class and [Num] or [Num, Num, Num]

    prob_fact = model_predict_proba(factual.to_frame().T)

    # Check how it's the output of multiple
    prob_fact_multiple = model_predict_proba(pd.concat([factual.to_frame().T, factual.to_frame().T]))

    if str(prob_fact).isnumeric():
        # Result returns a number directly

        if len(np.array(prob_fact_multiple).shape) == 1:
            # Single: Num
            # Multiple: [Num, Num, Num]
            def _mp1(x): return np.array([model_predict_proba(x)]) if x.shape[0] == 1 else \
                np.array(model_predict_proba(x))
        else:
            # Single: Num
            # Multiple: [[Num], [Num], [Num]]
            def _mp1(x): return np.array([model_predict_proba(x)]) if x.shape[0] == 1 else \
                np.array(model_predict_proba(x))[:, 0]

    elif len(np.array(prob_fact).shape) == 1:
        if len(np.array(prob_fact_multiple).shape) == 1:
            # Single: [Num]
            # Multiple [Num, Num, Num]
            def _mp1(x): return np.array(model_predict_proba(x))
        else:
            # Single: [Num]
            # Multiple [[Num], [Num], [Num]]
            if len(np.array(prob_fact_multiple)[0]) > 1:
                # If there are more than one class, the multiclass nonspecific strategy will be performed
                # where the factual (the largest initial output) is compared with the highest scoring class
                # TODO: This can be improved with other strategies like second best or even specific class
                _adjusted_nonspecific_mp1 = _adjust_multiclass_nonspecific(
                    np.array([factual.to_numpy()]), lambda z: np.array([model_predict_proba(z)])
                    if np.array(z).shape[0] == 1 else np.array(model_predict_proba(z)))

                def _mp1(x): return _adjusted_nonspecific_mp1(x)
            else:
                def _mp1(x): return np.array([model_predict_proba(x)[0]]) if x.shape[0] == 1 else \
                    np.array(model_predict_proba(x))[:, 0]
    else:
        # Single: [[Num]]
        # Multiple [[Num], [Num], [Num]]
        if len(prob_fact[0]) > 1:
            # If there are more than one class, the multiclass nonspecific strategy will be performed
            # where the factual (the largest initial output) is compared with the highest scoring class
            # TODO: This can be improved with other strategies like second best or even specific class
            _adjusted_nonspecific_mp1 = _adjust_multiclass_nonspecific(np.array([factual.to_numpy()]),
                                                                       lambda z: np.array(model_predict_proba(z)))

            def _mp1(x): return _adjusted_nonspecific_mp1(x)
        else:
            # This function gives an array containing the class 1 probability
            def _mp1(x): return np.array(model_predict_proba(x))[:, 0]

    return _mp1


def _adjust_model_class(factual, mp1):
    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    _mp1c = mp1
    # Adjust class, it must be binary and lower than 0
    if mp1(np.array([cf_try]))[0] > 0.5:
        def _mp1c(x): return 1 - mp1(x)

    return _mp1c


def _adjust_image_model(img, model_predict, segments, replace_img):
    # x is the array where the index represent the segmentation area number and the value 1 means the original
    # image and 0 the replaced image
    def _mic(seg_arr):
        converted_images = _seg_to_img(seg_arr, img, segments, replace_img)

        return model_predict(np.array(converted_images))

    return _mic


def _convert_to_numpy(data):
    if type(data) == pd.DataFrame:
        return data.to_numpy()
    if type(data) == np.ndarray:
        return data


def _adjust_multiclass_nonspecific(factual: np.ndarray, mic):
    # Compare the factual class value to the other highest
    pred_factual = mic(np.array([factual]) if len(factual.shape) == 1 else factual)
    factual_idx = np.argmax(pred_factual)

    def _mimns(cf_candidates):
        # Calculate the prediction of the candidates
        pred_cfs = mic(_convert_to_numpy(cf_candidates)).astype('float')

        # Get the value of the factual class
        pred_factual_class = np.copy(pred_cfs[:, factual_idx])
        # Now, to guarantee to get the best non factual value, let's consider the factual idx as -infinity
        pred_cfs[:, factual_idx] = -np.inf
        # Now, get the best values for each candidate
        pred_best_cf_class = np.max(pred_cfs, axis=1)

        # Make the comparison
        class_dif = pred_factual_class - pred_best_cf_class

        # Return a probability like value, where 0 is the factual and 1 counterfactual
        return 1/(1+np.e**class_dif)

    return _mimns


def _adjust_multiclass_second_best(factual: np.ndarray, mic):
    # In this function, we get the second-highest scored class and make it as the CF to be found.
    # Then, the score of this target (the originally second-highest scored class) is compared with the other best
    # result.

    # Compare the factual class value to the other highest
    pred_factual = mic(np.array([factual]) if len(factual.shape) == 1 else factual)
    factual_idx = copy.copy(np.argmax(pred_factual))
    pred_factual[0][factual_idx] = -np.inf
    cf_idx = copy.copy(np.argmax(pred_factual))

    def _mimns(cf_candidates):
        # Calculate the prediction of the candidates
        # Sometimes it can receive a dataframe, if it's the case, treat accordingly
        pred_cfs = mic(_convert_to_numpy(cf_candidates))

        # Get the value of the cf class
        pred_cf_class = np.copy(pred_cfs[:, cf_idx])
        # Now, to guarantee to get the best non cf value, let's consider the factual idx as -infinity
        pred_cfs[:, cf_idx] = -np.inf
        # Now, get the best value which is not the CF
        pred_best_ncf_class = np.max(pred_cfs, axis=1)

        # Make the comparison
        class_dif = pred_best_ncf_class - pred_cf_class

        # Return a probability like value, where 0 is the factual and 1 counterfactual
        return 1/(1+np.e**class_dif)

    return _mimns


def _adjust_textual_classifier(textual_classifier, converter, original_text_classification):
    return lambda array_texts: textual_classifier(converter(array_texts)) \
        if original_text_classification < 0.5 else 1 - textual_classifier(converter(array_texts))
