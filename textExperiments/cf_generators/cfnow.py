from cfnow.cf_finder import find_text


def cfnow_greedy(text_input, textual_classifier, factual_class, model):
    cf_results = find_text(text_input, textual_classifier, cf_strategy='greedy')

    cf_optimized = cf_results.cf
    cf_not_optimized = cf_results.cf_not_optimized

    cf_html_highlighted_optimized = cf_results.cf_html_highlight
    cf_html_highlighted_not_optimized = cf_results.cf_html_not_optimized

    cf_class_optimized = textual_classifier([cf_results.cf])[0][0]
    cf_class_not_optimized = textual_classifier([cf_results.cf_not_optimized])[0][0]

    cf_words_optimized = cf_results.cf_replaced_words
    cf_words_not_optimized = cf_results.cf_not_optimized_replaced_words

    cf_time_optimized = cf_results.time_cf
    cf_time_not_optimized = cf_results.time_cf_not_optimized

    return [cf_optimized, cf_html_highlighted_optimized, cf_class_optimized, cf_words_optimized, cf_time_optimized], \
           [cf_not_optimized, cf_html_highlighted_not_optimized, cf_class_not_optimized, cf_words_not_optimized,
            cf_time_not_optimized]


def cfnow_random(text_input, textual_classifier, factual_class, model):
    cf_results = find_text(text_input, textual_classifier, cf_strategy='random')

    cf_optimized = cf_results.cf
    cf_not_optimized = cf_results.cf_not_optimized

    cf_html_highlighted_optimized = cf_results.cf_html_highlight
    cf_html_highlighted_not_optimized = cf_results.cf_html_not_optimized

    cf_class_optimized = textual_classifier([cf_results.cf])[0][0]
    cf_class_not_optimized = textual_classifier([cf_results.cf_not_optimized])[0][0]

    cf_words_optimized = cf_results.cf_replaced_words
    cf_words_not_optimized = cf_results.cf_not_optimized_replaced_words

    cf_time_optimized = cf_results.time_cf
    cf_time_not_optimized = cf_results.time_cf_not_optimized

    return [cf_optimized, cf_html_highlighted_optimized, cf_class_optimized, cf_words_optimized, cf_time_optimized], \
           [cf_not_optimized, cf_html_highlighted_not_optimized, cf_class_not_optimized, cf_words_not_optimized,
            cf_time_not_optimized]
