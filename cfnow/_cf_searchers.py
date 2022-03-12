"""
This module has the functions used to find a CF explanation.
"""
import copy
from itertools import combinations
from collections import deque

import numpy as np

from ._data_standardizer import _ohe_detector, _get_ohe_list


def _random_generator(factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes, increase_threshold,
                      tabu_list, size_tabu, avoid_back_original, verbose):
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
    while cf_try_prob <= 0.5 and iterations < 5:
        for n_changes in range(1, 6):
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
            possible_changes = np.concatenate(
                [c for c in [changes_cat_bin, changes_cat_ohe, changes_num_up, changes_num_down] if len(c) > 0])

            # If the flag to back to original is set, then, remove all changes that make the result back to original values
            if avoid_back_original:
                n_same_cf_try = (cf_try == factual).sum()
                n_same_possible_changes = ((possible_changes + cf_try) == factual[0]).sum(axis=1)
                idx_same_drop = np.where(n_same_possible_changes >= n_same_cf_try)[0]
                possible_changes = np.delete(possible_changes, idx_same_drop, axis=0)

            # # New variables
            # n_changes = 1
            # limit_random_array = 10000

            # This is the number of combinations possible
            n_comb = np.prod([*range(len(factual) + 1 - n_changes, len(factual) + 1)])

            # If the number of combinations is lower than the threshold, create a list of modifications
            if n_comb < len(factual) * 2 and n_comb < 20000:
                index_comb = [x for x in combinations([*range(len(possible_changes))], n_changes)]
            elif len(factual) * 2 < 20000:
                index_comb = np.random.randint(0, len(possible_changes), (len(factual) * 2, n_changes))
            else:
                index_comb = np.random.randint(0, len(possible_changes), (20000, n_changes))

            # # If the number of combinations is lower than the threshold, create a list of modifications
            # if (n_comb) < 20000:
            #     index_comb = [x for x in combinations([*range(len(possible_changes))], n_changes)]
            # else:
            #     index_comb = np.random.randint(0, len(possible_changes), (50000, n_changes))

            # The if avoids OHE being incorrectly summed (we can not sum the change of two OHE in the same category)
            random_changes = np.array([np.sum(possible_changes[r, :], axis=0) for r in index_comb if
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
        # Increase momentum
        add_momentum += 1

    return cf_try


def _super_sedc(factual, mp1c, feat_types, it_max, ft_change_factor, ohe_list, ohe_indexes, increase_threshold,
                tabu_list, size_tabu, avoid_back_original, verbose):
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

    return cf_try
