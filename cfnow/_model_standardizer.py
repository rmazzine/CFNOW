"""
This module has the functions which take a model prediction function (which returns probability) and return
a standardized function that can be used by the CF generator.
"""
import copy

import numpy as np
import pandas as pd


def _standardize_predictor(factual, model_predict_proba):
    prob_fact = model_predict_proba(factual.to_frame().T)

    # Convert the output of prediction function to something that can be treated

    # Check how it's the output of multiple
    prob_fact_multiple = model_predict_proba(pd.concat([factual.to_frame().T, factual.to_frame().T]))

    # mp1 always return the 1 class and [Num] or [Num, Num, Num]
    if str(prob_fact).isnumeric():
        # Result returns a number directly

        if len(np.array(prob_fact_multiple).shape) == 1:
            # Single: Num
            # Multiple: [Num, Num, Num]
            mp1 = lambda x: np.array([model_predict_proba(x)]) if x.shape[0] == 1 else np.array(model_predict_proba(x))
        else:
            # Single: Num
            # Multiple: [[Num], [Num], [Num]]
            index_1 = 0
            if len(np.array(prob_fact_multiple)[0]) == 2:
                index_1 = 1
            # This function gives an array containing the class 1 probability
            mp1 = lambda x: np.array([model_predict_proba(x)]) if x.shape[0] == 1 else np.array(model_predict_proba(x))[
                                                                                       :, index_1]

    elif len(np.array(prob_fact).shape) == 1:
        if len(np.array(prob_fact_multiple).shape) == 1:
            # Single: [Num]
            # Multiple [Num, Num, Num]
            mp1 = lambda x: np.array(model_predict_proba(x))
        else:
            # Single: [Num]
            # Multiple [[Num], [Num], [Num]]
            index_1 = 0
            if len(np.array(prob_fact_multiple)[0]) == 2:
                index_1 = 1
            mp1 = lambda x: np.array(model_predict_proba(x))[:, index_1]
    else:
        # Single: [[Num]]
        # Multiple [[Num], [Num], [Num]]
        index_1 = 0
        if len(prob_fact[0]) == 2:
            index_1 = 1
        # This function gives an array containing the class 1 probability
        mp1 = lambda x: np.array(model_predict_proba(x))[:, index_1]

    return mp1


def _adjust_model_class(factual, mp1):
    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    mp1c = mp1
    # Adjust class, it must be binary and lower than 0
    if mp1(np.array([cf_try]))[0] > 0.5:
        mp1c = lambda x: 1 - mp1(x)

    return mp1c


def _adjust_image_model(img, model_predict, segments, replace_img):
    # x is the array where the index represent the segmentation area number and the value 1 means the original
    # image and 0 the replaced image
    return lambda x: model_predict(np.array([np.array([np.isin(segments, np.where(xr)[0]).astype(float)]*3).reshape(img.shape)*img + np.array([np.isin(segments, np.where(xr==0)[0]).astype(float)]*3).reshape(img.shape)*replace_img for xr in x]))


def _adjust_image_multiclass_nonspecific(factual, mic):
    # Compare the factual class value to the other highest
    pred_factual = mic([factual])
    factual_idx = np.argmax(pred_factual)

    def mimns(cf_candidates):
        # Calculate the prediction of the candidates
        # Sometimes it can receive a dataframe, if it's the case, treat accordingly
        if type(cf_candidates) == pd.DataFrame:
            pred_cfs = mic(cf_candidates.to_numpy())
        else:
            pred_cfs = mic(cf_candidates)
        # Get the value of the factual class
        pred_factual_class = np.copy(pred_cfs[: ,factual_idx])
        # Now, to guarantee to get the best non factual value, let's consider the factual idx as -infinity
        pred_cfs[: ,factual_idx] = -np.inf
        # Now, get the best values for each candidate
        pred_best_cf_class = np.max(pred_cfs, axis=1)

        # Make the comparison
        class_dif = pred_factual_class - pred_best_cf_class

        # Return a probability like value, where 1 is the factual and 0 counterfactual
        return 1/(1+np.e**(class_dif))

    return mimns
