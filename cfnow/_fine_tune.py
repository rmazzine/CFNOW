"""
This function has the finetune optimizers which take a CF and try to minimize the objective function while
keeping the CF as CF (i.e. not returning to the original class).
"""
import copy
import logging
import warnings
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd

from ._data_standardizer import _get_ohe_list
from ._obj_functions import _obj_manhattan


def _create_mod_change(changes_back_factual, change_idx, change_original_idx, ft_change_factor, _feat_idx_to_type):
    # Get the respective change vector
    # If numerical, use factor
    if _feat_idx_to_type(change_original_idx) == 'num':
        mod_change = changes_back_factual[change_idx] * ft_change_factor
    else:
        mod_change = changes_back_factual[change_idx]

    return mod_change


def _calculate_change_factor(c_cf, changes_back_factual, feat_distances, changes_back_original_idxs,  mp1c, c_cf_c):
    # Generate the CF back factual
    cf_back_factual = c_cf + np.array(copy.copy(changes_back_factual))

    # Now, calculate the probability for each different index
    cf_back_factual_probs = mp1c(cf_back_factual)

    prediction_dif = (c_cf_c - cf_back_factual_probs)

    # Calculate how much a unitary change cause a change in probability
    # The probability should be as negative as possible
    # The distance (objective function) should be as close to zero as possible (as it cannot be negative)
    # For a positive probability, we just multiply the prediction change by the distance
    # For a negative probability, we just divide the prediction change by the distance
    change_factor_feat = []
    for f_pc, f_d in zip(prediction_dif, feat_distances[changes_back_original_idxs]):
        if f_pc >= 0:
            change_factor_feat.append(f_pc*f_d)
        else:
            # Avoid division by zero
            if f_d == 0:
                change_factor_feat.append(0)
            else:
                change_factor_feat.append(f_pc/f_d)

    return np.array(change_factor_feat)


def _generate_change_vectors(factual, factual_np, c_cf, _feat_idx_to_type, tabu_list,
                             ohe_indexes, ohe_list, ft_threshold_distance):
    """
    This function takes the factual and CF and calculates:
    changes_back_factual: an array of possible modifications to return to original values
    changes_back_original_idxs: the indexes of features that have different values from original
    feat_distances: the distance of the features to the original feature values
    """
    # Generate change vector with all changes that would make the cf return
    # to factual for each feature
    changes_back_factual = []
    changes_back_original_idxs = []

    # Threshold of distance
    feat_distances = np.abs(c_cf - factual_np)

    # The features that very close to the factual (below of the threshold)
    # must be considered as zero
    feat_distances[np.where(feat_distances < ft_threshold_distance)] = 0

    # First, get indexes that are different
    diff_idxs = np.where(feat_distances != 0)

    # Create a flatten list of forbidden indexes
    forbidden_indexes = [item for sublist in tabu_list for item in sublist]

    template_vector = np.full((factual.shape[0],), 0, dtype=float)
    # For each different index
    for di in diff_idxs[0]:
        # if index is in Tabu list, skip this index
        if di in forbidden_indexes:
            continue
        # CHANGE OPERATIONS ALWAYS ADD TO THE CF VECTOR
        # THEN IF FACTUAL [0,1,10] AND CF IS [0,1,2] THEN
        # CHANGE VECTOR MUST BE [0, 0, 8]
        change_vector = copy.copy(template_vector)
        # If it belongs to num, then just create a vector that returns to original value
        if _feat_idx_to_type(di) == 'num':
            change_vector[di] = factual_np[di] - c_cf[di]
        elif di in ohe_indexes:
            change_ohe_idx = _get_ohe_list(di, ohe_list)
            change_vector[change_ohe_idx] = factual_np[change_ohe_idx] - c_cf[change_ohe_idx]
        else:
            # It's binary
            change_vector[di] = factual_np[di] - c_cf[di]

        changes_back_factual.append(change_vector)
        # Add original_index to track
        changes_back_original_idxs.append(di)

    return changes_back_factual, changes_back_original_idxs, feat_distances


def _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list):
    """
    This function has additional stop conditions for the optimization step
    """

    # Get all feature types
    all_feat_types = list(set(feat_types.values()))

    # Get len of OHE features
    len_ohe = len([item for sublist in ohe_list for item in sublist])

    # If all categorical and distance is 1, then, it's the best optimized solution, return result
    if len(all_feat_types) == 1 and all_feat_types[0] == 'cat':
        if _obj_manhattan(factual_np, c_cf) == 1:
            return True

    # If all OHE and the distance is 2, then, it's the best optimized solution, return result
    if len(all_feat_types) == 1 and len_ohe == len(factual_np):
        if _obj_manhattan(factual_np, c_cf) == 2:
            return True

    # Check time limit
    if (datetime.now() - time_start).total_seconds() >= limit_seconds:
        logging.log(20, 'Timeout reached')
        return True

    return False


def _fine_tuning(cf_data_type, factual, cf_out, mp1c, ohe_list, ohe_indexes, increase_threshold, feat_types,
                 ft_change_factor, it_max, size_tabu, ft_it_max, ft_threshold_distance, time_start, limit_seconds,
                 cf_finder, avoid_back_original, threshold_changes, verbose):
    """

    :param cf_data_type: Type of data
    :param factual: Factual point
    :param cf_out: Counterfactual generated from factual
    :param mp1c: Predictor function wrapped in a predictable way
    :param ohe_list: List of OHE features
    :param ohe_indexes: List of OHE indexes
    :param increase_threshold: The threshold in score which, if not higher,
    will count to activate Tabu and Momentum
    :param feat_types: The type for each feature
    :param ft_change_factor: Proportion of numerical feature to be used in their modification
    :param it_max: Maximum number of iterations
    :param size_tabu: Size of Tabu list
    :param ft_it_max: Maximum number of iterations for finetune
    :param ft_threshold_distance: A threshold to identify if further modifications are (or not) effective
    :param time_start: Initial start which algorithm started
    :param limit_seconds: Time limit for CF generation and optimization
    :param cf_finder: The function used to find CF explanations
    :param avoid_back_original: If active, does not allow features going back to their original values for
    the greedy strategy
    :param verbose: Gives additional information about the process if true
    :return: An optimized counterfactual
    """
    # Maps feat idx to name to type
    feat_idx_to_name = pd.Series(factual.index).to_dict()
    def _feat_idx_to_type(x): return feat_types[feat_idx_to_name[x]]

    factual_np = factual.to_numpy()

    tabu_list = deque(maxlen=size_tabu)

    # Create array to store the best solution
    # It has: the VALID CF, the CF score and the objective function  (L1 distance)
    best_solution = [copy.copy(cf_out), mp1c(np.array([cf_out]))[0], _obj_manhattan(factual_np, cf_out)]

    # Create variable to store current solution - FIRST TIME
    c_cf = copy.copy(cf_out)

    # Check classification
    c_cf_c = mp1c(np.array([c_cf]))[0]

    for i in range(ft_it_max):

        if _stop_optimization_conditions(factual_np, c_cf, limit_seconds, time_start, feat_types, ohe_list):
            return best_solution

        if verbose:
            logging.log(10, f'Fine tuning: Prob = {c_cf_c}\n'
                            f'Distance = {_obj_manhattan(factual_np, c_cf)}\n'
                            f'Tabu list elements = {tabu_list}')

        changes_back_factual, changes_back_original_idxs, feat_distances = _generate_change_vectors(
            factual, factual_np, c_cf, _feat_idx_to_type, tabu_list, ohe_indexes, ohe_list, ft_threshold_distance)

        # In some situations, the Tabu list contains all modified indexes, in this case
        # give some inspiration to explore new regions, now, currently there's no implementation of that
        # If list of changes is 0, return result
        if len(changes_back_factual) == 0:
            warnings.warn('Change list is empty')
            return best_solution

        change_factor_feat = _calculate_change_factor(c_cf, changes_back_factual, feat_distances,
                                                      changes_back_original_idxs,  mp1c, c_cf_c)

        # Select the change based on the lowest change factor
        change_idx = np.argmin(change_factor_feat)
        change_original_idx = changes_back_original_idxs[change_idx]

        mod_change = _create_mod_change(changes_back_factual, change_idx, change_original_idx,
                                        ft_change_factor, _feat_idx_to_type)

        # Make modification
        c_cf = c_cf + mod_change

        # Check classification
        c_cf_c = mp1c(np.array([c_cf]))[0]

        # Check if still a cf
        if c_cf_c > 0.5:
            # Calculate objective function
            c_cf_o = _obj_manhattan(factual_np, c_cf)
            # Check if it's a better solution
            if c_cf_o < best_solution[2]:
                best_solution = [copy.copy(c_cf), c_cf_c, c_cf_o]
        else:
            # Not a CF
            # Add index to Tabu list
            # If numerical or binary, just add the single index
            # However, if it's OHE add all related indexes
            if change_original_idx in ohe_indexes:
                tabu_list.append(_get_ohe_list(change_original_idx, ohe_list))
            else:
                tabu_list.append([change_original_idx])

            # Return to CF, however, considering the Tabu list
            c_cf = cf_finder(cf_data_type=cf_data_type,
                             factual=pd.DataFrame([c_cf], columns=factual.index).iloc[0],
                             mp1c=mp1c,
                             feat_types=feat_types,
                             it_max=it_max,
                             ft_change_factor=ft_change_factor,
                             ohe_list=ohe_list,
                             ohe_indexes=ohe_indexes,
                             tabu_list=tabu_list,
                             size_tabu=size_tabu,
                             increase_threshold=increase_threshold,
                             avoid_back_original=avoid_back_original,
                             ft_time=time_start,
                             ft_time_limit=limit_seconds,
                             threshold_changes=threshold_changes,
                             verbose=verbose)

    return best_solution
