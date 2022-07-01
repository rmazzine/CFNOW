"""
This module has the functions used to find a CF explanation.

* For Categorical (binary) there's only one change, flipping 0->1 or 1->0

* For Categorical (OHE) there's a complex change, which considers flipping two binaries

* For Numerical we can increase or decrease the feature according
to ft_change_factor and momentum (feature*ft_change_factor + momentum)
"""
import math
import copy
from collections import deque
from datetime import datetime
from itertools import combinations

import numpy as np

from ._data_standardizer import _ohe_detector, _get_ohe_list


def _random_generator(cf_data_type, factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes,
                      increase_threshold, tabu_list, size_tabu, avoid_back_original, ft_time, ft_time_limit, verbose):
    """
    This algorithm takes a random strategy to find a minimal set of changes which change the classification prediction

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
    :param verbose: Gives additional information about the process if true
    :return: A counterfactual or the best try to achieve it
    """

    threshold_changes = 2000
    if cf_data_type == 'tabular':
        threshold_changes = 2000
    if cf_data_type == 'image':
        threshold_changes = 200
    if cf_data_type == 'text':
        threshold_changes = 500

    # Additional momentum to avoid being stuck in a minimum, starts with zero, however, if Tabu list is activated
    # and changes are not big, activate it
    add_momentum = 0

    # If tabu_list is None, then, not consider it assigning an empty list
    if tabu_list is None:
        tabu_list = deque(maxlen=(size_tabu))

    # Define the cf try
    cf_try = copy.copy(factual).to_numpy()

    # Identify the indexes of categorical and numerical variables
    indexes_cat = np.where(np.isin(factual.index, [c for c, t in feat_types.items() if t == 'cat']))[0]
    indexes_num = sorted(list(set([*range(len(factual))]) - set(indexes_cat.tolist())))

    # Create identity matrix for each type of variable
    arr_changes_cat_bin = np.eye(len(factual))[list(set(indexes_cat) - set(ohe_indexes))]
    arr_changes_cat_ohe = np.eye(len(factual))
    arr_changes_num = np.eye(len(factual))[indexes_num]

    iterations = 1
    cf_try_prob = mp1c(factual.to_frame().T)[0]

    # Repeat until max iterations or a CF is found
    while cf_try_prob <= 0.5 and iterations < it_max:
        # Each iteration will run from 1 to total number of features
        for n_changes in range(1, len(ohe_list)+len(indexes_num)+arr_changes_cat_bin.shape[0]):

            # Create changes
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

            # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_up = arr_changes_num * ft_change_factor * cf_try + arr_changes_num * add_momentum
            # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_down = -copy.copy(changes_num_up)

            def _count_subarray(sa):
                return sum([len(s) for s in sa])

            # Length for each kind of change
            len_ccb = len(changes_cat_bin)
            len_cco = len(changes_cat_ohe)
            len_cnu = len(changes_num_up)
            len_cnd = len(changes_num_up)

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
                    [_count_subarray(ohe_list[0:x]), _count_subarray(ohe_list[0:x])+_count_subarray(ohe_list[x:x+1])]
                for x in range(len(ohe_list))}
            num_placeholders = [f'num_{x}' for x in range(len(changes_num_up))]
            num_placeholder_to_change_idx = {f'num_{x}': x for x in range(len(changes_num_up))}

            # All possible changes
            change_feat_options = pc_idx_bin+num_placeholders+ohe_placeholders

            # Calculate the number of possible change combinations
            n_comb_base = math.comb(len(change_feat_options), n_changes)

            # The number of possible modifications can be larger than the previous calculated, since for each OHE
            # and numerical feature, there are more than one possible change (for OHE depends on the feature values and
            # for numerical can be up or down). Therefore, if the number of combinations is below the threshold,
            # these situations must be checked to have a precise calculation of the possible modifications number.
            if n_comb_base <= threshold_changes:
                idx_comb_changes = [*combinations(change_feat_options, n_changes)]
                corrected_num_changes = 0
                for icc in idx_comb_changes:
                    comb_rows = [1]
                    for ohp in ohe_placeholders:
                        if ohp in icc:
                            idx_ohe_min, idx_ohe_max = ohe_placeholder_to_change_idx[ohp]
                            comb_rows.append(idx_ohe_max - idx_ohe_min)
                    for nmp in num_placeholders:
                        if nmp in icc:
                            comb_rows.append(2)
                    corrected_num_changes += np.prod(comb_rows)
            else:
                # The sample will be 1 above the limit threshold
                corrected_num_changes = threshold_changes + 1

            if corrected_num_changes <= threshold_changes:
                # In this case, there are few modifications, so we will calculate all combinations

                # Calculate all possible combinations
                idx_comb_changes = [*combinations(change_feat_options, n_changes)]

                # Now, each time a OHE feature is modified, consider all possible modifications
                update_placeholder = [list(icc) for icc in idx_comb_changes]
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

                changes_idx = update_placeholder

            else:
                # If the number of possible changes is larger than the threshold, then make a sample (threshold+1)
                # of possible modifications

                # This array will receive the modifications
                changes_idx = []

                # Some suggested changes can be invalid because they are repeated or select the same OHE and numerical
                # features two times, therefore, we define the variable below to give chances to repeat the generation
                # process trying to get a different change set
                tries_gen = 1

                # While the number of changes is not equal to sample size and the number of tries
                # (to generate unique change sets) was not reached
                while len(changes_idx) < threshold_changes and tries_gen < threshold_changes*2:
                    # TODO: Change the choice process to not take the same numeric of OHE feature two times
                    sample_features = np.random.choice(change_feat_options, n_changes)

                    # OHE and binary cannot be selected two times
                    # Then, considering the set of changes without num_, the set must be the same size of the list
                    sample_features_not_num = [sf for sf in sample_features if 'num_' not in str(sf)]
                    if len(set(sample_features_not_num)) != len(sample_features_not_num):
                        continue

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
                    set_change_idx_row = set(change_idx_row)
                    tries_gen += 1
                    if set_change_idx_row not in changes_idx:
                        changes_idx.append(set_change_idx_row)

                changes_idx = [[int(ci) for ci in c] for c in changes_idx]

            # If the number of changes is 0, skip to next iteration
            if len(changes_idx) == 0:
                continue

            # Below ensures OHE will not be incorrectly summed
            # (we cannot sum the change of two OHE in the same category)
            random_changes = np.array([np.sum(possible_changes[r, :], axis=0) for r in changes_idx if
                                       sum([_ohe_detector(r, ic) for ic in ohe_list]) == 0])

            # If there are no random changes, return best result
            if len(random_changes) == 0:
                return cf_try

            # if the Tabu list is larger than zero
            if len(tabu_list) > 0:
                # Remove all rows which the sum of absolute change vector partition is larger than zero
                forbidden_indexes = [item for sublist in tabu_list for item in sublist]
                idx_to_remove = np.where(np.abs(random_changes[:, forbidden_indexes]) != 0)[0]
                random_changes = np.delete(random_changes, idx_to_remove, axis=0)

            # If, after removing, there's no changes, return CF
            if len(random_changes) == 0:
                return cf_try

            # Create array with CF candidates
            cf_candidates = cf_try + random_changes

            # Calculate probabilities
            prob_cf_candidates = mp1c(cf_candidates)

            # Identify which index had the best performance towards objective, it will take the first best
            best_arg = np.argmax(prob_cf_candidates)

            # Update CF try
            cf_try = cf_try + random_changes[best_arg]

            # Calculate how much the score got better
            score_increase = [-cf_try_prob]

            # Update the score
            cf_try_prob = mp1c(np.array([cf_try]))[0]

            # Basic verbose report
            if verbose:
                print(f'SEDC probability {cf_try_prob}')

            # Calculate how much the score got better
            score_increase.append(cf_try_prob)

            # If the score changed, break the loop
            if cf_try_prob >= 0.5:
                break

        # After a full iteration over all features, increase momentum
        add_momentum += 1

        # Count an iteration
        iterations += 1

        # Check time for fine-tune
        if ft_time is not None:
            if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
                return cf_try
    return cf_try


def _super_sedc(cf_data_type, factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes,
                increase_threshold, tabu_list, size_tabu, avoid_back_original, ft_time, ft_time_limit, verbose):
    """
        This algorithm makes sequential changes which will better increase the score to find a CF

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

    # Identify the indexes of categorical and numerical variables
    indexes_cat = np.where(np.isin(factual.index, [c for c, t in feat_types.items() if t == 'cat']))[0]
    indexes_num = sorted(list(set([*range(len(factual))]) - set(indexes_cat.tolist())))

    # Create identity matrix for each type of variable
    arr_changes_cat_bin = np.eye(len(factual))[list(set(indexes_cat) - set(ohe_indexes))]
    arr_changes_cat_ohe = np.eye(len(factual))
    arr_changes_num = np.eye(len(factual))[indexes_num]

    iterations = 1

    cf_try_prob = mp1c(factual.to_frame().T)[0]

    # Implement a threshold for score increase, this avoids having useless moves
    # Before entering to the loop, define it being larger than the threshold
    score_increase = increase_threshold + 1

    # Repeat until max iterations
    # The third condition (score threshold) should only be applied if the Tabu is not activated
    # since the activation of Tabu can lead to decrease in score, and it's normal
    while cf_try_prob <= 0.5 and iterations < it_max and \
            ((score_increase >= increase_threshold) if not activate_tabu else True):

        # Make changes

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

        # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
        changes_num_up = arr_changes_num * ft_change_factor * cf_try + arr_changes_num * add_momentum
        # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
        changes_num_down = -copy.copy(changes_num_up)

        # Create changes array
        changes = np.concatenate(
            [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

        # If the flag to back to original is set, then, remove all changes that make the result back to original values
        if avoid_back_original:
            n_same_cf_try = (cf_try == factual).sum()
            n_same_changes = ((changes+cf_try) == factual[0]).sum(axis=1)
            idx_same_drop = np.where(n_same_changes >= n_same_cf_try)[0]
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

        # If no changes, return cf
        if len(changes) == 0:
            return cf_try

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

        # Basic verbose report
        if verbose:
            print(f'SEDC probability {cf_try_prob}')

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
                add_momentum = 0

        # Update number of tries
        iterations += 1

        # Check time for fine-tune
        if ft_time is not None:
            if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
                return cf_try

    return cf_try
