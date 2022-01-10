"""
This module has functions which gets information about the factual data and standardize to the CF generator.
"""
from collections import defaultdict


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


def _ohe_detector(lst1, lst2):
    return len(set(lst1).intersection(lst2)) > 1


def _get_ohe_list(f_idx, ohe_list):
    for ol in ohe_list:
        if f_idx in ol:
            return ol
