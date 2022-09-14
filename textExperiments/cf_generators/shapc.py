import time
import copy

import numpy as np
import pandas as pd

import shap

from textExperiments.cf_generators.common import _text_to_token_vector, _convert_change_vectors_func, \
    _adjust_textual_classifier


def make_exp_shapc(text_input, textual_classifier, factual_class, model):
    text_words, change_vector, text_replace = _text_to_token_vector(text_input)
    vector_words = [w for w in text_replace if len(w) > 0]
    converter = _convert_change_vectors_func(text_words, change_vector, text_replace, False)
    converter_highlight_html = _convert_change_vectors_func(text_words, change_vector, text_replace, True)
    textual_classifier_shap = _adjust_textual_classifier(textual_classifier, converter, factual_class[0][0])
    textual_classifier_shap(change_vector)

    def custom_masker(mask, x):
        return (x * mask).reshape(1, len(x))

    init_time = time.time()

    explainer = shap.Explainer(textual_classifier_shap, masker=custom_masker)
    shap_explanation = explainer(change_vector.to_numpy(), max_evals=10000)

    total_time = time.time() - init_time

    shap_values = pd.Series(shap_explanation.values[0] * -1).sort_values(ascending=False)
    shap_values_idx_ordered = list(shap_values.index)

    factual_shap_class = textual_classifier_shap(change_vector)[0][0]
    for max_m_idx in range(len(shap_values_idx_ordered[:100])):
        m_idx_list = shap_values_idx_ordered[:max_m_idx]
        test_cf = copy.copy(change_vector.to_numpy())
        test_cf[0][m_idx_list] = 0
        prob = textual_classifier_shap(test_cf)

        if (prob[0] > 0.5) != (factual_shap_class > 0.5):
            cf_class = textual_classifier(np.array(converter(test_cf)))[0][0]
            replace_words = [vector_words[m_idx][0] for m_idx in m_idx_list]
            return [[converter(test_cf)[0], converter_highlight_html(test_cf)[0], cf_class, replace_words, total_time]]

    return [[None, None, None, None, None]]
