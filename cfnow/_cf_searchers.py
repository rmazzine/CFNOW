"""
This module has the functions used to find a CF explanation.
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

    threshold_changes = 2000
    if cf_data_type == 'tabular':
        threshold_changes = 2000
    if cf_data_type == 'image':
        threshold_changes = 200
    if cf_data_type == 'text':
        threshold_changes = 500

    recent_improvements = deque(maxlen=(3))

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
    while cf_try_prob <= 0.5 and iterations < it_max:
        for n_changes in range(1, len(ohe_list)+len(indexes_num)+arr_changes_cat_bin.shape[0]):
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

            # For numerical up - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_up = arr_changes_num * ft_change_factor * cf_try + arr_changes_num * add_momentum
            # For numerical down - HERE, NUMBERS WHICH ARE ZERO WILL REMAIN ZERO
            changes_num_down = -copy.copy(changes_num_up)

            def count_subarray(sa):
                return sum([len(s) for s in sa])

            # Length for each kind of change
            len_ccb = len(changes_cat_bin)
            len_cco = len(changes_cat_ohe)
            len_cnu = len(changes_num_up)
            len_cnd = len(changes_num_up)

            # Create changes array
            possible_changes = np.concatenate(
                [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

            comb_idx = []
            # Possible changes index
            pc_idx_bin = [*range(len_ccb)]
            pc_idx_ohe = [*range(len_ccb, len_ccb + len_cco)]
            pc_idx_nup = [*range(len_ccb + len_cco, len_ccb + len_cco + len_cnu)]
            pc_idx_ndw = [*range(len_ccb + len_cco + len_cnu, len_ccb + len_cco + len_cnu + len_cnd)]

            # If less than the threshold, calculate all possibilities
            idx_comb_changes = []
            ohe_placeholders = [f'ohe_{x}' for x in range(len(ohe_list))]
            ohe_placeholder_to_change_idx = {f'ohe_{x}': [count_subarray(ohe_list[0:x]), count_subarray(ohe_list[0:x])+ count_subarray(ohe_list[x:x+1])]  for x in range(len(ohe_list))}
            num_placeholders = [f'num_{x}' for x in range(len(changes_num_up))]
            num_placeholder_to_change_idx = {f'num_{x}': x for x in range(len(changes_num_up))}

            change_feat_options = pc_idx_bin+num_placeholders+ohe_placeholders

            n_comb_base = math.comb(len(change_feat_options), n_changes)

            # If the base is larger than 2000, then, the corrected will be larger than 2000
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
                # Calculate the sample
                corrected_num_changes = threshold_changes + 1

            if corrected_num_changes <= threshold_changes:
                # There are few modifications, calculate all combinations
                idx_comb_changes = [*combinations(change_feat_options, n_changes)]

                # Now, fix OHE placeholders
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
                # Make a sample of the possible modifications
                changes_idx = []
                tries_gen = 1
                while len(changes_idx) < threshold_changes and tries_gen < threshold_changes*2:
                    sample_features = np.random.choice(change_feat_options, n_changes)
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

            # Skip to next iteration
            if len(changes_idx) == 0:
                continue

            # # If the number of combinations is lower than the threshold, create a list of modifications
            # if (n_comb) < 20000:
            #     index_comb = [x for x in combinations([*range(len(possible_changes))], n_changes)]
            # else:
            #     index_comb = np.random.randint(0, len(possible_changes), (50000, n_changes))

            # The if avoids OHE being incorrectly summed (we can not sum the change of two OHE in the same category)
            random_changes = np.array([np.sum(possible_changes[r, :], axis=0) for r in changes_idx if
                                       sum([_ohe_detector(r, ic) for ic in ohe_list]) == 0])

            # If there are no random changes, return best result
            if len(random_changes) == 0:
                return cf_try

            # if the Tabu list is larger than zero
            if len(tabu_list) > 0:
                # Remove all rows which the sum of absolute change vector
                # partition is larger than zero

                # Flatten indexes
                forbidden_indexes = [item for sublist in tabu_list for item in sublist]
                idx_to_remove = np.where(np.abs(random_changes[:, forbidden_indexes]) != 0)[0]
                random_changes = np.delete(random_changes, idx_to_remove, axis=0)

            # If after removing, there's no changes, return
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
            if verbose:
                print(f'SEDC probability {cf_try_prob}')

            # Calculate how much the score got better
            score_increase.append(cf_try_prob)

            # The change array will be mixed randomly from 1 to n
            # Create a zero matrix
            # However, allow all combinations up to a number
            # Create a random matrix with numbers up to the number of columns  np.random.randint(0,10, (10,10))
            # Add the row offset for each column
            # Those are the indexes that must be replaced by 1

            # This can be obtained with a multiplication

            # Then apply

            # search for any CF

            # If yes return and do not try again, since it will give a higher modified CF

            # This random generation can happen several times

            # If the score changed, break the loop
            if cf_try_prob >= 0.5:
                break
        # Increase momentum
        add_momentum += 1
        # Count an iteration
        iterations += 1
        # Check time for fine tuning
        if ft_time is not None:
            if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
                return cf_try
    return cf_try


def _super_sedc(cf_data_type, factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes,
                increase_threshold, tabu_list, size_tabu, avoid_back_original, ft_time, ft_time_limit, verbose):
    recent_improvements = deque(maxlen=(3))

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
    # The third condition (limit threshold) should only be applied if the Tabu is not activated
    # since the activation of Tabu can lead to decrease in score and it's normal
    while cf_try_prob <= 0.5 and iterations < it_max and ((score_increase >= increase_threshold) if not activate_tabu else True):
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
        if verbose:
            print(f'SEDC probability {cf_try_prob} -{np.where(changes[best_arg] != 0)[0][0]}')
            print(cf_try)

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
        # Check time for fine tuning
        if ft_time is not None:
            if (datetime.now() - ft_time).total_seconds() >= ft_time_limit:
                return cf_try

    return cf_try
