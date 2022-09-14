import re
import copy

import nltk
import numpy as np
import pandas as pd


def _text_to_token_vector(text):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Then get phrase tokens, replace ' with \\ to preserve contracted words
    text_words = nltk.word_tokenize(text.replace("'", "\\"))
    # Go back and replace \\ to '
    text_words = [w.replace("\\", "'") for w in text_words]

    # Create a substitution list for all identifiable words (excluding the special characters)
    text_replace_word = [[w, ''] if len(re.sub('[^A-Za-z0-9]+', '', w)) > 0 else [] for w in text_words]

    # This is a dictionary which keys are the original word position and values are the possible replacement
    change_vector_links = {idx: [*range(len(subs))] for idx, subs in enumerate(text_replace_word) if len(subs) > 0}

    # Now create the DataFrame representation
    # Create factual row values
    # Create column names
    factual_row = [1] * len(change_vector_links)
    factual_col = [f'{k}' for k in change_vector_links.keys()]

    change_vector = pd.DataFrame([dict(zip(factual_col, factual_row))])

    return text_words, change_vector, text_replace_word


def _convert_change_vectors_func(text_words, change_vector, text_antonyms, highlight_html):
    return lambda input_change_vector: _change_vector_to_text(input_change_vector, text_words, change_vector,
                                                              text_antonyms, highlight_html)


def _adjust_textual_classifier(textual_classifier, converter, original_text_classification):
    return lambda array_texts: textual_classifier(
        converter(array_texts)) if original_text_classification < 0.5 else 1 - textual_classifier(
        converter(array_texts))


def _change_vector_to_text(input_change_vector, text_words, change_vector, text_antonyms, highlight_html):
    n_rows = len(input_change_vector)
    out_texts = [copy.copy(text_words) for _ in range(n_rows)]

    # It can be a pandas DataFrame or numpy
    if type(input_change_vector) == pd.DataFrame:
        change_coordinates = list(input_change_vector.iloc[0][input_change_vector.iloc[0] == 0].index)
        change_coordinates = [[0] * len(change_coordinates), change_coordinates]
    else:
        change_coordinates = list(np.where(input_change_vector == 0))
        change_coordinates[1] = [change_vector.columns[ci] for ci in change_coordinates[1]]

    # Loop over the possible column changes
    for idx_cv, cc in enumerate(change_coordinates[0]):
        i_text, idx_w = change_coordinates[0][idx_cv], change_coordinates[1][idx_cv]

        if not highlight_html:
            # If it's in the list, it means it is equal to zero and must be removed (replaced by '')
            out_texts[i_text][int(idx_w)] = ''
        else:
            out_texts[i_text][int(idx_w)] = f'<span style="color: red;font-weight: bold; ' \
                                            f'text-decoration: line-through;">{out_texts[i_text][int(idx_w)]}</span>'

    out_full_texts = [_untokenize(t) for t in out_texts]

    return out_full_texts


def _untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
