"""
This module has the functions used to find a CF explanation.

* For Categorical (binary) there's only one change, flipping 0->1 or 1->0

* For Categorical (OHE) there's a complex change, which considers flipping two binaries

* For Numerical we can increase or decrease the feature according
to ft_change_factor and momentum (feature*ft_change_factor + momentum)
"""
import math
import copy
import logging
import operator
from collections import deque
from datetime import datetime
from functools import reduce
from itertools import combinations

import numpy as np

from ._data_standardizer import _ohe_detector, _get_ohe_list


def _create_change_matrix(factual, feat_types, ohe_indexes):
    # Identify the indexes of categorical and numerical variables
    indexes_cat = np.where(np.isin(factual.index, [c for c, t in feat_types.items() if t == 'cat']))[0]
    indexes_num = sorted(list({*range(len(factual))} - set(indexes_cat.tolist())))

    # Create identity matrix for each type of variable
    arr_changes_cat_bin = np.eye(len(factual))[list(set(indexes_cat) - set(ohe_indexes))]
    # For OHE we take all indexes because we need the row make reference to the column changed
    arr_changes_cat_ohe = np.eye(len(factual))
    arr_changes_num = np.eye(len(factual))[indexes_num]

    return indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num


def _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe):
    # For categorical ohe
    changes_cat_ohe_list = []
    for ohe_group in ohe_list:
        changes_cat_ohe_list.append(
            arr_changes_cat_ohe[ohe_group] - (arr_changes_cat_ohe[ohe_group] * cf_try).sum(axis=0))
    if len(changes_cat_ohe_list) > 0:
        changes_cat_ohe = np.concatenate(changes_cat_ohe_list)
    else:
        changes_cat_ohe = []

    return changes_cat_ohe_list, changes_cat_ohe


def _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,  arr_changes_num,
                            arr_changes_cat_bin, arr_changes_cat_ohe):
    # Create changes
    # For categorical binary
    changes_cat_bin = arr_changes_cat_bin * (1 - 2 * cf_try)
    # Changes for OHE
    changes_cat_ohe_list, changes_cat_ohe = _create_ohe_changes(cf_try, ohe_list, arr_changes_cat_ohe)
    # The array below is intended to label positive feature numbers as 1 and negative as -1
    # This has the utility for momentum since for negative numbers, the momentum must be negative
    num_direction_change = (cf_try > 0).astype(float)-(cf_try < 0).astype(float)
    # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
    changes_num_up = arr_changes_num * ft_change_factor * cf_try + num_direction_change*arr_changes_num * add_momentum
    # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
    changes_num_down = -copy.copy(changes_num_up)

    return changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down


def _count_subarray(sa):
    return sum([len(s) for s in sa])


def _replace_ohe_placeholders(update_placeholder, ohe_placeholders, ohe_placeholder_to_change_idx, pc_idx_ohe):
    # This function replace the OHE placeholders (like 'ohe_1') to actual changes
    for ohp in ohe_placeholders:
        updating_placeholder = []
        # For each index combination changes in the update placeholder variable
        for icc in update_placeholder:
            if ohp in icc:
                ohp_indexes = ohe_placeholder_to_change_idx[ohp]
                icc_ohp_index = icc.index(ohp)
                for ohp_idx in pc_idx_ohe[ohp_indexes[0]:ohp_indexes[1]]:
                    base_icc_ohp = copy.copy(icc)
                    base_icc_ohp[icc_ohp_index] = ohp_idx
                    updating_placeholder.append(base_icc_ohp)
            else:
                updating_placeholder.append(icc)
        # Update for next iteration
        update_placeholder = updating_placeholder

    return update_placeholder


def _replace_num_placeholders(update_placeholder, num_placeholders, num_placeholder_to_change_idx,
                              pc_idx_nup, pc_idx_ndw):
    # This function replace the numerical placeholders (like 'num_1') to actual changes

    # For each numerical feature changed, add the two possible modifications (increase and decrease)
    for nmp in num_placeholders:
        updating_placeholder = []
        # For each index combination changes in the update placeholder variable
        for icc in update_placeholder:
            if nmp in icc:
                nmp_index = num_placeholder_to_change_idx[nmp]
                icc_nmp_index = icc.index(nmp)

                icc_nmp_up = copy.copy(icc)
                icc_nmp_dw = copy.copy(icc)

                icc_nmp_up[icc_nmp_index] = pc_idx_nup[nmp_index]
                icc_nmp_dw[icc_nmp_index] = pc_idx_ndw[nmp_index]

                updating_placeholder.extend([icc_nmp_up, icc_nmp_dw])
            else:
                updating_placeholder.append(icc)
        # Update for next iteration
        update_placeholder = updating_placeholder

    return update_placeholder


def _generate_random_changes_all_possibilities(n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders,
                                               change_feat_options, ohe_placeholders, ohe_placeholder_to_change_idx,
                                               num_placeholder_to_change_idx):
    # This function generates all possible (and valid) combination between features.
    # This is not a straightforward problem since OHE and numeric (up and down) changes have some specificities:
    # For OHE: You cannot select a same category OHE in a single change. For Numeric: Similar as selecting UP and DOWN
    # would result in an 0 change

    # Calculate all possible combinations
    idx_comb_changes = [*combinations(change_feat_options, n_changes)]

    # Now, each time a OHE feature is modified, consider all possible modifications
    update_placeholder = [list(icc) for icc in idx_comb_changes]

    # Replace OHE placeholders
    update_placeholder = _replace_ohe_placeholders(update_placeholder, ohe_placeholders,
                                                   ohe_placeholder_to_change_idx, pc_idx_ohe)
    # Replace numerical placeholders
    changes_idx = _replace_num_placeholders(update_placeholder, num_placeholders, num_placeholder_to_change_idx,
                                            pc_idx_nup, pc_idx_ndw)

    return changes_idx


def _create_random_changes(sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
                           num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw):
    # This function creates, randomly, changes based on numerical and OHE placeholders (since it can have more than
    # a single modification). For binary, since there's only one possible change, we just append the modification.

    # Create modifications for OHE, numerical and binary features
    change_idx_row = []
    for sf in sample_features:
        if sf in ohe_placeholders:
            ohp_indexes = ohe_placeholder_to_change_idx[sf]
            change_idx_row.append(np.random.choice(pc_idx_ohe[ohp_indexes[0]:ohp_indexes[1]], 1)[0])
        elif sf in num_placeholders:
            nmp_index = num_placeholder_to_change_idx[sf]
            change_idx_row.append(np.random.choice([pc_idx_nup[nmp_index], pc_idx_ndw[nmp_index]]))
        else:
            change_idx_row.append(sf)

    return set(change_idx_row)


def _generate_random_changes_sample_possibilities(n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders,
                                                  change_feat_options, ohe_placeholders, ohe_placeholder_to_change_idx,
                                                  num_placeholder_to_change_idx, threshold_changes):
    # If the number of possible changes is larger than the threshold, then make a sample (threshold+1)
    # of possible modifications

    # This array will receive the modifications
    changes_idx = []

    # Some suggested changes can be invalid because they are repeated or select the same OHE and numerical
    # features two times, therefore, we define the variable below to give chances to repeat the generation
    # process trying to get a different change set
    tries_gen = 0

    # While the number of changes is not equal to sample size and the number of tries
    # (to generate unique change sets) was not reached
    while len(changes_idx) < threshold_changes and tries_gen < threshold_changes * 2:
        tries_gen += 1

        # TODO: Change the choice process to not take the same numeric of OHE feature two times
        sample_features = np.random.choice(change_feat_options, n_changes)

        # OHE and binary cannot be selected two times
        # Then, considering the set of changes without num_, the set must be the same size of the list
        # This allows numerical features to be added (or subtracted) several times
        sample_features_not_num = [sf for sf in sample_features if 'num_' not in str(sf)]
        if len(set(sample_features_not_num)) != len(sample_features_not_num):
            continue

        set_change_idx_row = _create_random_changes(
            sample_features, ohe_placeholders, num_placeholders, ohe_placeholder_to_change_idx,
            num_placeholder_to_change_idx, pc_idx_ohe, pc_idx_nup, pc_idx_ndw)

        if set_change_idx_row not in changes_idx:
            changes_idx.append(set_change_idx_row)

    changes_idx = [[int(ci) for ci in c] for c in changes_idx]

    return changes_idx


def _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders, ohe_placeholder_to_change_idx,
                               threshold_changes, n_changes):
    # Calculate the number of possible change combinations
    n_comb_base = math.comb(len(change_feat_options), n_changes)

    # The number of possible modifications can be larger than the previous calculated (n_comb_base), since for each OHE
    # and numerical feature, there are more than one possible change (for OHE depends on the feature values and
    # for numerical can be up or down). Therefore, if the number of combinations initially calculated (n_comb_base)
    # is below the threshold, another calculation must be done to get the corrected number of possible
    # changes (corrected_num_changes) considering the OHE and numeric features previously mentioned.
    # Example:
    # - Two binary features (b1, b2): There's only 1 possible combination = (b1, b2)
    # - One binary (b1) and one OHE (o1) with three possible values (o1_1, o1_2, o1_3): There are 3 possible
    # combinations = (b1, o1_1), (b1, o1_2), (b1, o1_3)
    # - One OHE (o1) with three possible values (o1_1, o1_2, o1_3) and one numerical (n1) with up and
    # down (n1_u, n1_d): There are 6 possible combinations: (o1_1, n1_u), (o1_1, n1_d), (o1_2, n1_u), (o1_2, n1_d),
    # (o1_3, n1_u), (o1_3, n1_d)
    if n_comb_base <= threshold_changes:
        # First we get all possible combinations with the features for a certain number of n_changes
        idx_comb_changes = [*combinations(change_feat_options, n_changes)]
        corrected_num_changes = 0
        for icc in idx_comb_changes:
            # Initially we consider there's only one possibility of combination: this will only
            # happen for two binary features
            comb_rows = [1]
            for ohp in ohe_placeholders:
                # For each OHE feature, we add to the comb_rows the number of possible values
                if ohp in icc:
                    idx_ohe_min, idx_ohe_max = ohe_placeholder_to_change_idx[ohp]
                    comb_rows.append(idx_ohe_max - idx_ohe_min)
            for nmp in num_placeholders:
                if nmp in icc:
                    # For each numeric feature, we consider 2 possible cases: up and down
                    comb_rows.append(2)
            # Then, after scanning all features in a single change (icc), we calculate the number of
            # derived modifications by multiplying the items inside comb_rows
            corrected_num_changes += reduce(operator.mul, comb_rows)
    else:
        # The sample will be 1 above the limit threshold
        corrected_num_changes = threshold_changes + 1

    return corrected_num_changes


def _generate_random_changes(changes_cat_bin, changes_cat_ohe, changes_num_up,
                             changes_num_down, ohe_list, n_changes, threshold_changes):
    # Length for each kind of change
    len_ccb = len(changes_cat_bin)
    len_cco = len(changes_cat_ohe)
    len_cnu = len(changes_num_up)
    len_cnd = len(changes_num_down)

    # Create changes array
    possible_changes = np.concatenate(
        [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

    # Possible changes index
    pc_idx_bin = [*range(len_ccb)]
    pc_idx_ohe = [*range(len_ccb, len_ccb + len_cco)]
    pc_idx_nup = [*range(len_ccb + len_cco, len_ccb + len_cco + len_cnu)]
    pc_idx_ndw = [*range(len_ccb + len_cco + len_cnu, len_ccb + len_cco + len_cnu + len_cnd)]

    # OHE and numerical are special types of feature change since for a same feature, they have multiple
    # possible modifications, then, placeholder codes are created to identify them (binary is not necessary
    # since the change is only one possible 1 to 0 or 0 to 1)
    ohe_placeholders = [f'ohe_{x}' for x in range(len(ohe_list))]
    ohe_placeholder_to_change_idx = {
        f'ohe_{x}':
            [_count_subarray(ohe_list[0:x]), _count_subarray(ohe_list[0:x]) + _count_subarray(ohe_list[x:x + 1])]
        for x in range(len(ohe_list))}
    num_placeholders = [f'num_{x}' for x in range(len(changes_num_up))]
    num_placeholder_to_change_idx = {f'num_{x}': x for x in range(len(changes_num_up))}

    # All possible changes
    change_feat_options = pc_idx_bin + num_placeholders + ohe_placeholders

    corrected_num_changes = _calc_num_possible_changes(change_feat_options, num_placeholders, ohe_placeholders,
                                                       ohe_placeholder_to_change_idx, threshold_changes, n_changes)

    if corrected_num_changes <= threshold_changes:
        changes_idx = _generate_random_changes_all_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx)

    else:
        changes_idx = _generate_random_changes_sample_possibilities(
            n_changes, pc_idx_ohe, pc_idx_nup, pc_idx_ndw, num_placeholders, change_feat_options, ohe_placeholders,
            ohe_placeholder_to_change_idx, num_placeholder_to_change_idx, threshold_changes)

    return possible_changes, changes_idx


def _random_generator_stop_conditions(iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf):

    if len(cf_unique) >= count_cf:
        return False

    if iterations >= it_max:
        return False

    # Check time for fine-tune
    if ft_time is not None:
        if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
            return False

    return True


def _random_generator(finder_strategy, cf_data_type, factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list,
                      ohe_indexes, increase_threshold, tabu_list, size_tabu, avoid_back_original, ft_time,
                      ft_time_limit, threshold_changes, count_cf, cf_unique, verbose):
    """
    This algorithm takes a random strategy to find a minimal set of changes which change the classification prediction

    :param finder_strategy: The strategy adopted by the CF generator
    :param cf_data_type: Type of data
    :param factual: The factual point
    :param mp1c: The predictor function wrapped in a predictable way
    :param feat_types: The type for each feature
    :param it_max: Maximum number of iterations
    :param ft_change_factor: Proportion of numerical feature to be used in their modification
    :param ohe_list: List of OHE features
    :param ohe_indexes: List of OHE indexes
    :param increase_threshold: NOT USED FOR RANDOM STRATEGY, since the momentum increase happens after one full
    run over features
    :param tabu_list: List of features forbidden to be modified
    :param size_tabu: Size of Tabu list
    :param avoid_back_original: NOT USED FOR RANDOM STRATEGY
    :param ft_time: If it's in the fine-tune process, tells the current fine-tune time
    :param ft_time_limit: The time limit for fine-tune
    :param count_cf: Number of CFs generated
    :param cf_unique: List of unique CFs generated
    :param verbose: Gives additional information about the process if true
    :return: A counterfactual or the best try to achieve it
    """

    # Additional momentum to avoid being stuck in a minimum, starts with zero, however, if Tabu list is activated
    # and changes are not big, activate it
    add_momentum = 0

    # If tabu_list is None, then, not consider it assigning an empty list
    if tabu_list is None:
        tabu_list = deque(maxlen=(size_tabu))

    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    # Get the indexes of cat and num features and also the matrix guiding the changes for each feature type
    indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = _create_change_matrix(
        factual, feat_types, ohe_indexes)

    iterations = 1
    cf_try_prob = mp1c(factual.to_frame().T)[0]

    # Repeat until max iterations or a CF is found
    while _random_generator_stop_conditions(iterations=iterations, cf_unique=cf_unique, ft_time=ft_time, it_max=it_max,
                                            ft_time_limit=ft_time_limit, count_cf=count_cf):
        # Each iteration will run from 1 to total number of features
        for n_changes in range(1, len(ohe_list)+len(indexes_num)+arr_changes_cat_bin.shape[0]):

            # Create changes to be applied in the factual instance
            changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
                _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,
                                        arr_changes_num, arr_changes_cat_bin, arr_changes_cat_ohe)

            # TODO: Add a memory to avoid investigating the same feature several times
            # Generate the modification vectors with the possible modifications
            possible_changes, changes_idx = _generate_random_changes(
                changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down, ohe_list,
                n_changes, threshold_changes)

            # Below ensures OHE will not be incorrectly summed
            # (we cannot sum the change of two OHE in the same category)
            random_changes = np.array([np.sum(possible_changes[r, :], axis=0) for r in changes_idx if
                                       sum([_ohe_detector(r, ic) for ic in ohe_list]) == 0])

            # if there are random changes and the Tabu list is larger than zero
            if (len(random_changes) > 0) and (len(tabu_list) > 0):
                # Remove all rows which the sum of absolute change vector partition is larger than zero
                forbidden_indexes = [item for sublist in tabu_list for item in sublist]
                idx_to_remove = np.where(np.abs(random_changes[:, forbidden_indexes]) != 0)[0]
                random_changes = np.delete(random_changes, idx_to_remove, axis=0)

            # Continue if there's no random change
            if len(random_changes) == 0:
                continue

            # Create array with CF candidates
            cf_candidates = cf_try + random_changes

            if len(cf_unique) > 0:
                cf_candidates_unique = set(map(tuple, cf_candidates)).difference(set(map(tuple, cf_unique)))
                cf_candidates = np.array(list(cf_candidates_unique))

            # Continue if there's no random change
            if len(cf_candidates) == 0:
                continue

            # Calculate probabilities
            prob_cf_candidates = mp1c(cf_candidates)
            best_arg = np.argmax(prob_cf_candidates)

            # Get CF arrays sorted by probability
            mask_cf = prob_cf_candidates >= 0.5
            cf_arrays = cf_candidates[mask_cf, :]
            cf_prob = prob_cf_candidates[mask_cf]
            cf_arrays = cf_arrays[cf_prob.argsort()[::-1]]

            # Add unique CFs
            cf_unique.extend(cf_arrays)

            if finder_strategy == 'sequential':
                # The random sequential strategy updates the CF try with the best candidate
                # Update CF try
                cf_try = cf_candidates[best_arg]
                # Update the score
                cf_try_prob = mp1c(np.array([cf_try]))[0]
            if finder_strategy is None:
                if len(cf_unique) > 0:
                    cf_try_prob = mp1c(np.array([cf_unique[0]]))[0]

            # Basic verbose report
            if verbose:
                logging.log(10, f'Best CF try probability = {cf_try_prob}')

            # If enough CFs break the loop
            if len(cf_unique) >= count_cf:
                break

        # After a full iteration over all features, increase momentum
        add_momentum += 1

        # Count an iteration
        iterations += 1

    return cf_unique


def _greedy_generator_stop_conditions(
        iterations, cf_unique, ft_time, it_max, ft_time_limit, count_cf, score_increase,
        increase_threshold, activate_tabu):

    if len(cf_unique) >= count_cf:
        return False

    if iterations >= it_max:
        return False

    if (score_increase < increase_threshold) if not activate_tabu else False:
        return False

    # Check time for fine-tune
    if ft_time is not None:
        if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
            return False

    return True


def _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe, changes_num_up,
                             changes_num_down, avoid_back_original):

    # Create changes array
    changes = np.concatenate(
        [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

    # If the flag to back to original is set, then, remove all changes that make the result back to original values
    if avoid_back_original:
        # The values that are equal to the original can remain the same,
        # however, modified values cannot be back to original, this creates the following logical table:
        #
        # True => Equal to original / False => Not equal to original.
        #
        # Current Counterfactual | Modified Counterfactual | Allowed? (interpretation
        #        True            |        False            |    YES (the feature was modified)
        #        True            |        True             |    YES (the feature was not modified and remain the same)
        #        False           |         True            |    NO  (the feature was modified back to original)
        #        False           |         False           |    YES (the feature was modified and still modified)
        #
        # Then, the following logical expression is used
        # AND( XOR(same_current_cf, modified_cf), modified_cf)
        same_current_cf = (cf_try == factual).to_numpy()
        modified_cf = ((changes+cf_try) == factual.to_numpy())
        value_back_to_original = np.logical_and(np.logical_xor(same_current_cf, modified_cf), modified_cf)
        idx_same_drop = np.where(value_back_to_original.sum(axis=1))[0]
        changes = np.delete(changes, idx_same_drop, axis=0)

    # Drop all zero rows
    changes = np.delete(changes, np.where(np.abs(changes).sum(axis=1) == 0)[0], axis=0)

    # if the Tabu list is larger than zero
    if len(tabu_list) > 0:
        # Remove all rows which the sum of absolute change vector
        # partition is larger than zero

        # Flatten indexes
        forbidden_indexes = [item for sublist in tabu_list for item in sublist]
        idx_to_remove = np.where(np.abs(changes[:, forbidden_indexes]) != 0)[0]
        changes = np.delete(changes, list(set(idx_to_remove)), axis=0)

    return changes


def _greedy_generator(
        finder_strategy, cf_data_type, factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list,
        ohe_indexes, increase_threshold, tabu_list, size_tabu, avoid_back_original, ft_time,
        ft_time_limit, threshold_changes, count_cf, cf_unique, verbose):
    """
        This algorithm makes sequential changes which will better increase the score to find a CF

        :param finder_strategy: The strategy adopted by the CF generator
        :param cf_data_type: NOT USED FOR GREEDY STRATEGY
        :param factual: The factual point
        :param mp1c: The predictor function wrapped in a predictable way
        :param feat_types: The type for each feature
        :param it_max: Maximum number of iterations
        :param ft_change_factor: Proportion of numerical feature to be used in their modification
        :param ohe_list: List of OHE features
        :param ohe_indexes: List of OHE indexes
        :param increase_threshold: The threshold in score which, if not higher,
        will count to activate Tabu and Momentum
        :param tabu_list: List of features forbidden to be modified
        :param size_tabu: Size of Tabu list
        :param avoid_back_original: If active, does not allow features going back to their original values
        :param ft_time: If it's in the fine-tune process, tells the current fine-tune time
        :param ft_time_limit: The time limit for fine-tune
        :param verbose: Gives additional information about the process if true
        :return: A counterfactual or the best try to achieve it
        """

    # Deque list to track if recent improvements were enough (larger than the increase_threshold)
    recent_improvements = deque(maxlen=(3))

    # Start with a greedy optimization, however, if the changes selected are the same and the score increase
    # is not good, start Tabu list
    activate_tabu = False

    # Additional momentum to avoid being stuck in a minimum, starts with zero, however, if Tabu list is activated
    # and changes are not big, activate it
    add_momentum = 0

    # If tabu_list is None, then, not consider it assigning an empty list
    if tabu_list is None:
        tabu_list = deque(maxlen=(size_tabu))

    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    # Get the indexes of cat and num features and also the matrix guiding the changes for each feature type
    indexes_cat, indexes_num, arr_changes_cat_bin, arr_changes_cat_ohe, arr_changes_num = _create_change_matrix(
        factual, feat_types, ohe_indexes)

    iterations = 1

    cf_try_prob = mp1c(factual.to_frame().T)[0]

    # Implement a threshold for score increase, this avoids having useless moves
    # Before entering to the loop, define it being larger than the threshold
    score_increase = increase_threshold + 1

    # Repeat until max iterations
    # The third condition (score threshold) should only be applied if the Tabu is not activated
    # since the activation of Tabu can lead to decrease in score, and it's normal
    while _greedy_generator_stop_conditions(
            iterations=iterations, cf_unique=cf_unique, ft_time=ft_time, it_max=it_max, ft_time_limit=ft_time_limit,
            count_cf=count_cf, score_increase=score_increase, increase_threshold=increase_threshold,
            activate_tabu=activate_tabu):

        # Create changes to be applied in the factual instance
        changes_cat_bin, changes_cat_ohe_list, changes_cat_ohe, changes_num_up, changes_num_down = \
            _create_factual_changes(cf_try, ohe_list, ft_change_factor, add_momentum,
                                    arr_changes_num, arr_changes_cat_bin, arr_changes_cat_ohe)

        changes = _generate_greedy_changes(factual, cf_try, tabu_list, changes_cat_bin, changes_cat_ohe,
                                           changes_num_up, changes_num_down, avoid_back_original)

        # If no changes, return cf
        if len(changes) == 0:
            continue

        # Create array with CF candidates
        cf_candidates = cf_try + changes

        if len(cf_unique) > 0:
            cf_candidates_unique = set(map(tuple, cf_candidates)).difference(set(map(tuple, cf_unique)))
            cf_candidates = np.array(list(cf_candidates_unique))

        # Continue if there's no random change
        if len(cf_candidates) == 0:
            continue

        # Calculate probabilities
        prob_cf_candidates = mp1c(cf_candidates)
        # Identify which index had the best performance towards objective, it will take the first best
        best_arg = np.argmax(prob_cf_candidates)

        # Get CF arrays sorted by probability
        mask_cf = prob_cf_candidates >= 0.5
        cf_arrays = cf_candidates[mask_cf, :]
        cf_prob = prob_cf_candidates[mask_cf]
        cf_arrays = cf_arrays[cf_prob.argsort()[::-1]]

        # Add unique CFs
        cf_unique.extend(cf_arrays)

        # Update CF try
        cf_try = cf_candidates[best_arg]

        # Calculate how much the score got better
        score_increase = [-cf_try_prob]

        # Update the score
        cf_try_prob = mp1c(np.array([cf_try]))[0]

        # Basic verbose report
        if verbose:
            logging.log(10, f'Best CF try probability = {cf_try_prob}')

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
            else:
                # If the improvement was larger than the threshold, the momentum is removed
                add_momentum = 0

        # Update number of tries
        iterations += 1

    return cf_unique
