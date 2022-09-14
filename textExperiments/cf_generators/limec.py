import time
import copy

import numpy as np
import tensorflow as tf

from lime.lime_text import LimeTextExplainer


def make_exp_limec(text_input, textual_classifier, factual_class, model):

    def textual_classifier_limec(input_texts):
        shape_input_text = np.array(input_texts).shape
        if shape_input_text[0] > 500:
            results = []
            for z in range(0, len(input_texts), 500):
                text_batch = input_texts[z:z + 500]
                results.append(tf.sigmoid(model(tf.constant(text_batch))).numpy())
            concat_results = np.concatenate(results)
            return np.concatenate([1 - concat_results, concat_results], axis=1)
        results = tf.sigmoid(model(tf.constant(input_texts))).numpy()
        return np.concatenate([1 - results, results], axis=1)

    explainer = LimeTextExplainer(class_names=[0, 1], bow=False)

    init_time = time.time()

    exp = explainer.explain_instance(text_input, textual_classifier_limec, num_features=100000)

    total_time = time.time() - init_time

    # Map the vocab index to the full text index
    map_full_lime_vocab = {}
    mal_full_lime_vocab_start = 0
    for v_idx, v in enumerate(exp.domain_mapper.indexed_string.inverse_vocab):
        for r_idx, r in enumerate(exp.domain_mapper.indexed_string.as_list[mal_full_lime_vocab_start:]):
            if v == r:
                map_full_lime_vocab[v_idx] = r_idx + mal_full_lime_vocab_start
                mal_full_lime_vocab_start += r_idx
                break

    # Words must be removed according the index
    fulltext_lime = copy.copy(exp.domain_mapper.indexed_string.as_list)
    fulltext_highlight_lime = copy.copy(exp.domain_mapper.indexed_string.as_list)
    for max_m_idx in range(len(exp.local_exp[1])):
        m_idx_list = [map_full_lime_vocab[m_idx] for m_idx, _ in exp.local_exp[1][:max_m_idx + 1]]
        clf_text = np.array([''.join(np.delete(fulltext_lime, m_idx_list))])
        prob = textual_classifier(clf_text)
        if (prob[0] > 0.5) != (factual_class[0] > 0.5):
            for m_idx in m_idx_list:
                fulltext_highlight_lime[m_idx] = f'<span style="color: red;font-weight: bold; ' \
                                                 f'text-decoration: line-through;">{fulltext_highlight_lime[m_idx]}' \
                                                 f'</span>'
            highlighted_cf_text = ''.join(fulltext_highlight_lime)
            removed_words = [exp.domain_mapper.indexed_string.as_list[m_idx] for m_idx in m_idx_list]

            return [[clf_text[0], highlighted_cf_text, prob[0][0], removed_words, total_time]]

    return [[None, None, None, None, None]]
