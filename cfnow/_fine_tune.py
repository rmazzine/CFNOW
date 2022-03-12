"""
This function has the fine-tune optimizers which take a CF and try to minimize the objective function while
keeping the CF as CF (i.e. not returning to the original class).
"""
import copy
import warnings
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd

from ._data_standardizer import _get_ohe_list
from ._obj_functions import _obj_manhattan


def _fine_tuning(factual, cf_out, mp1c, ohe_list, ohe_indexes, increase_threshold, feat_types, ft_change_factor,
                 it_max, size_tabu, ft_it_max, ft_threshold_distance, time_start, limit_seconds, cf_finder,
                 avoid_back_original, verbose):
    feat_idx_to_name = pd.Series(factual.index).to_dict()
    feat_idx_to_type = lambda x: feat_types[feat_idx_to_name[x]]

    factual_np = factual.to_numpy()

    tabu_list = deque(maxlen=size_tabu)

    # Get all feature types
    all_feat_types = list(set(feat_types.values()))

    # Create array to store the best solution
    # It has: the VALID CF, the CF score and the objective function  (L1 distance)
    best_solution = [copy.copy(cf_out), mp1c(np.array([cf_out]))[0], _obj_manhattan(factual_np, cf_out)]

    # Create variable to store current solution - FIRST TIME
    c_cf = copy.copy(cf_out)

    # Check classification
    c_cf_c = mp1c(np.array([c_cf]))[0]

    for i in range(ft_it_max):

        # If all categorical and distance is 1, then, it's the best optimized solution, return result
        if len(all_feat_types) == 1 and all_feat_types[0] == 'cat':
            if _obj_manhattan(factual_np, c_cf) == 1:
                return best_solution

        # Check time limit
        if (datetime.now() - time_start).total_seconds() >= limit_seconds:
            print('Timeout reached')
            return best_solution

        if verbose:
            print(tabu_list)
            print(f'Fine tuning: Prob={c_cf_c} / Distance={_obj_manhattan(factual_np, c_cf)}')

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
            if feat_idx_to_type(di) == 'num':
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

        # In some situations, the Tabu list contains all modified indexes, in this case
        # give some inspiration to explore new regions, now, currently there's no implementation of that
        # If list of changes is 0, return result
        if len(changes_back_factual) == 0:
            warnings.warn('Change list is empty')
            return best_solution

        # Generate the CF back factual
        cf_back_factual = c_cf + np.array(copy.copy(changes_back_factual))

        # Now, calculate the probability for each different index
        cf_back_factual_probs = mp1c(cf_back_factual)

        # Calculate how much an unitary change cause a change in probability
        change_factor_feat = (c_cf_c - cf_back_factual_probs) / feat_distances[changes_back_original_idxs]

        # Select the change based on the lowest change factor
        change_idx = np.argmin(change_factor_feat)
        change_original_idx = changes_back_original_idxs[change_idx]

        # Get the respective change vector
        # If numerical, use factor
        if feat_idx_to_type(change_original_idx) == 'num':
            mod_change = changes_back_factual[change_idx] * ft_change_factor
        else:
            mod_change = changes_back_factual[change_idx]

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
            # Add index to Tabu list
            # If numerical or binary, just add the single index
            # However, if it's OHE add all related indexes
            if change_original_idx in ohe_indexes:
                tabu_list.append(_get_ohe_list(change_original_idx, ohe_list))
            else:
                tabu_list.append([change_original_idx])

            # Return to CF, however, considering the Tabu list
            c_cf = cf_finder(factual=pd.DataFrame([c_cf], columns=factual.index).iloc[0],
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
                             verbose=verbose)

    return best_solution
