import copy
import warnings
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


def _check_factual(factual):
    # Factual must be a pandas Series
    try:
        assert type(factual) == pd.Series
    except AssertionError:
        raise TypeError(f'Factual must be a Pandas Series. However it is {type(factual)}.')


def _check_vars(factual, feat_types):
    # The number of feat_types must be the same as the number of factual features
    try:
        missing_var = list(set(factual.index) - set(feat_types.keys()))
        extra_var = list(set(feat_types.keys()) - set(factual.index))
        assert len(missing_var) == 0 and len(extra_var) == 0
    except AssertionError:
        if len(missing_var) > 0 and len(extra_var) > 0:
            raise AssertionError(f"\nThe features:\n {','.join(missing_var)}\nmust have their type defined in feat_types.\
                                 \n\nAnd the features:\n {','.join(extra_var)}\nare not defined in the factual point")
        elif len(missing_var) > 0:
            raise AssertionError(
                f"The features:\n {','.join(missing_var)}\nmust have their type defined in feat_types.")
        elif len(extra_var) > 0:
            raise AssertionError(f"The features:\n {','.join(extra_var)}\nare not defined in the factual point.")


def check_prob_func(factual, model_predict_proba):
    # Test model function and get the classification of factual
    try:
        prob_fact = model_predict_proba(factual.to_frame().T)
    except Exception as err:
        raise Exception('Error when using the model_predict_proba function.')


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


def _get_ohe_params(factual, has_ohe):
    ohe_list = []
    ohe_indexes = []
    # if has_ohe:
    if has_ohe:
        prefix_to_class = defaultdict(list)
        for col_idx, col_name in enumerate(factual.index):
            col_split = col_name.split('_')
            if len(col_split) > 1:
                prefix_to_class[col_split[0]].append(col_idx)

        ohe_list = [idx_list for _, idx_list in prefix_to_class.items() if len(idx_list) > 1]
        ohe_indexes = [item for sublist in ohe_list for item in sublist]

    return ohe_list, ohe_indexes


def _adjust_model_class(factual, mp1):
    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    mp1c = mp1
    # Adjust class, it must be binary and lower than 0
    if mp1(np.array([cf_try]))[0] > 0.5:
        mp1c = lambda x: 1 - mp1(x)

    return mp1c


def _super_sedc(factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes, increase_threshold,
                tabu_list, size_tabu, hard_try, verbose):
    recent_improvements = deque(maxlen=(5))

    # Start with a greedy optimization, however, if the changes selected are the same and the score increase
    # is not good, start Tabu list
    activate_tabu = False

    # Additional momentum to avoid being stuck in a minimum, starts with zero, however, if Tabu list is activated
    # and changes are not big, activate it
    add_momentum = 0

    # If tabu_list is None, then, disconsider it assigning an empty list
    if tabu_list is None:
        tabu_list = deque(maxlen=(size_tabu))

    # For Categorical (binary) there's only one change, flipping 0->1 or 1->0
    # For Categorical (OHE) there's a complex change, which considers
    # flipping two binaries
    # For Numerical we can increase 50% of input or decrease 50%

    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    # Identify the indexes of categorical and numerical variables
    indexes_cat = np.where(np.isin(factual.index, [c for c, t in feat_types.items() if t == 'cat']))[0]
    indexes_num = sorted(list(set([*range(len(factual))]) - set(indexes_cat.tolist())))

    # Create identity matrixes for each type of variable
    arr_changes_cat_bin = np.eye(len(factual))[list(set(indexes_cat) - set(ohe_indexes))]
    arr_changes_cat_ohe = np.eye(len(factual))
    arr_changes_num = np.eye(len(factual))[indexes_num]

    iterations = 1
    cf_try_prob = mp1c(factual.to_frame().T)[0]
    # Implement a threshold for score increase, this avoids having useless moves
    score_increase = increase_threshold + 1
    # Repeat until max iterations
    while cf_try_prob <= 0.5 and iterations < it_max and score_increase >= increase_threshold:
        # Changes
        # For categorical binary
        changes_cat_bin = arr_changes_cat_bin * (1 - 2 * cf_try)
        # For categorical ohe
        changes_cat_ohe_list = []
        for ohe_group in ohe_list:
            changes_cat_ohe_list.append(
                arr_changes_cat_ohe[ohe_group] - (arr_changes_cat_ohe[ohe_group] * cf_try).sum(axis=0))
        if len(changes_cat_ohe_list) > 0:
            changes_cat_ohe = np.concatenate(changes_cat_ohe_list)
        else:
            changes_cat_ohe = []
        if not hard_try:
            # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_up = arr_changes_num * ft_change_factor * cf_try + arr_changes_num * add_momentum
            # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_down = -copy.copy(changes_num_up)
        else:
            # The hard try is necessary for cases which the numerical results are very low (close to 0)
            # and then, they need an extra momentum to find CF explanations
            # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_up = arr_changes_num * 1 * copy.copy(factual).to_numpy() + arr_changes_num
            # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_down = -copy.copy(changes_num_up)

        # Create changes array
        changes = np.concatenate(
            [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

        # if the Tabu list is larger than zero
        if len(tabu_list) > 0:
            # Remove all rows which the sum of absolute change vector
            # partition is larger than zero

            # Flatten indexes
            forbidden_indexes = [item for sublist in tabu_list for item in sublist]
            idx_to_remove = np.where(np.abs(changes[:, forbidden_indexes]) != 0)[0]
            changes = np.delete(changes, idx_to_remove, axis=0)

        # Create array with CF candidates
        cf_candidates = cf_try + changes

        # Calculate probabilities
        prob_cf_candidates = mp1c(cf_candidates)

        # Identify which index had the best performance towards objective, it will take the first best
        best_arg = np.argmax(prob_cf_candidates)

        # Update CF try
        cf_try = cf_try + changes[best_arg]

        # Calculate how much the score got better
        score_increase = [-cf_try_prob]

        # Update the score
        cf_try_prob = mp1c(np.array([cf_try]))[0]
        if verbose:
            print(
                f'{"HARD-TRY" if hard_try else ""}SEDC probability {cf_try_prob} -{np.where(changes[best_arg] != 0)[0][0]}')

        # Calculate how much the score got better
        score_increase.append(cf_try_prob)
        score_increase = sum(score_increase)

        # If the sum of recent improvements is lower than 0.001 and all changes are in the same class
        # possibly we are stuck in a local minimum, then, activate Tabu optimization
        recent_improvements.append(score_increase)
        if sum(recent_improvements) < 0.001:
            activate_tabu = True

        # Stuck measure 1
        # If tabu is activated, include changed index to the list
        if activate_tabu:
            # It can be the first detected change since, for numerical results, the change will be the modified
            # index itself, for binary too. However, it's probably not true for OHE, however, since we use the
            # function "_get_ohe_list", it does not matter, since any index from OHE will return the list of OHE
            # which will be forbidden
            first_detected_change = np.where(changes[best_arg] != 0)[0][0]
            if first_detected_change in ohe_indexes:
                tabu_list.append(_get_ohe_list(first_detected_change, ohe_list))
            else:
                tabu_list.append([first_detected_change])

            # Stuck measure 2
            if sum(recent_improvements) < 0.001:
                add_momentum += 1

        # Update number of tries
        iterations += 1

    return cf_try


def _obj_function(factual_np, c_cf):
    return sum(np.abs(factual_np - c_cf))


def _get_ohe_list(f_idx, ohe_list):
    for ol in ohe_list:
        if f_idx in ol:
            return ol


def _fine_tunning(factual, cf_out, mp1c, ohe_list, ohe_indexes, increase_threshold, feat_types, ft_change_factor,
                  it_max, size_tabu, ft_it_max, ft_threshold_distance, time_start, limit_seconds, hard_try, verbose):
    feat_idx_to_name = pd.Series(factual.index).to_dict()
    feat_idx_to_type = lambda x: feat_types[feat_idx_to_name[x]]

    factual_np = factual.to_numpy()

    tabu_list = deque(maxlen=(size_tabu))

    # Create array to store the best solution
    # It has: the VALID CF, the CF score and the objective function  (L1 distance)
    best_solution = [copy.copy(cf_out), mp1c(np.array([cf_out]))[0], _obj_function(factual_np, cf_out)]

    # Create variable to store current solution - FIRST TIME
    c_cf = copy.copy(cf_out)

    # Check classification
    c_cf_c = mp1c(np.array([c_cf]))[0]

    for i in range(ft_it_max):
        # Check time limit
        if (datetime.now() - time_start).total_seconds() >= limit_seconds:
            print('Timeout reached')
            break

        if verbose:
            print(f'Fine tuning: Prob={c_cf_c} / Distance={_obj_function(factual_np, c_cf)}')

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

        template_vector = np.full((factual.shape[0],), 0, dtype=(float))
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
            c_cf_o = _obj_function(factual_np, c_cf)
            # Check if it's a better solution
            if c_cf_o < best_solution[2]:
                best_solution = [copy.copy(cf_out), c_cf_c, c_cf_o]
        else:
            # Add index to Tabu list
            # If numerical or binary, just add the single index
            # However, if it's OHE add all related indexes
            if change_original_idx in ohe_indexes:
                tabu_list.append(_get_ohe_list(change_original_idx, ohe_list))
            else:
                tabu_list.append([change_original_idx])

            # Return to CF, however, considering the Tabu list
            c_cf = _super_sedc(factual=pd.Series(c_cf),
                               mp1c=mp1c,
                               feat_types=feat_types,
                               it_max=it_max,
                               ft_change_factor=ft_change_factor,
                               ohe_list=ohe_list,
                               ohe_indexes=ohe_indexes,
                               tabu_list=tabu_list,
                               size_tabu=size_tabu,
                               increase_threshold=increase_threshold,
                               hard_try=hard_try,
                               verbose=verbose)

    return best_solution


def findcf(factual, feat_types, model_predict_proba,
           increase_threshold=0.001, it_max=1000, limit_seconds=120, ft_change_factor=0.5, ft_it_max=50, size_tabu=5,
           ft_threshold_distance=0.01, has_ohe=False, verbose=False):
    # If Tabu size list is larger than the number of features issue a warning and reduce to size_features - 1
    if len(factual) < size_tabu:
        size_tabu_new = len(factual) - 1
        warnings.warn(f'The number of features ({len(factual)}) is lower than the Tabu list size ({size_tabu}),'
                      f'then, we reduced to the number of features minus 1 (={size_tabu_new})')
        size_tabu = size_tabu_new

    # Timer now
    time_start = datetime.now()

    # Make checks
    _check_factual(factual)
    _check_vars(factual, feat_types)
    check_prob_func(factual, model_predict_proba)

    # Generate standardized predictor
    mp1 = _standardize_predictor(factual, model_predict_proba)

    # Correct class
    mp1c = _adjust_model_class(factual, mp1)

    # Generate OHE parameters if it has OHE variables
    ohe_list, ohe_indexes = _get_ohe_params(factual, has_ohe)

    hard_try = False

    # Generate CF using SEDC
    cf_out = _super_sedc(factual=factual,
                         mp1c=mp1c,
                         feat_types=feat_types,
                         it_max=it_max,
                         ft_change_factor=ft_change_factor,
                         ohe_list=ohe_list,
                         ohe_indexes=ohe_indexes,
                         increase_threshold=increase_threshold,
                         tabu_list=None,
                         size_tabu=size_tabu,
                         hard_try=hard_try,
                         verbose=verbose)

    # Check if CF was found
    if mp1c(np.array([cf_out]))[0] < 0.5:
        # Try again however, with hard reduction factor
        hard_try = True
        cf_out = _super_sedc(factual=factual,
                             mp1c=mp1c,
                             feat_types=feat_types,
                             it_max=it_max,
                             ft_change_factor=ft_change_factor,
                             ohe_list=ohe_list,
                             ohe_indexes=ohe_indexes,
                             increase_threshold=increase_threshold,
                             tabu_list=None,
                             size_tabu=size_tabu,
                             hard_try=hard_try,
                             verbose=verbose)

    if mp1c(np.array([cf_out]))[0] < 0.5:
        raise Warning('Test')

    # Fine tune the counterfactual
    cf_out_ft = _fine_tunning(factual=factual,
                              cf_out=cf_out,
                              mp1c=mp1c,
                              ohe_list=ohe_list,
                              ohe_indexes=ohe_indexes,
                              increase_threshold=increase_threshold,
                              feat_types=feat_types,
                              ft_change_factor=ft_change_factor,
                              it_max=it_max,
                              size_tabu=size_tabu,
                              ft_it_max=ft_it_max,
                              ft_threshold_distance=ft_threshold_distance,
                              time_start=time_start,
                              limit_seconds=limit_seconds,
                              hard_try=hard_try,
                              verbose=verbose)

    return cf_out_ft
