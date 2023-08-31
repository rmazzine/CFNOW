"""
This module has functions which gets information about the factual data and standardize to the CF generator.
"""
import os
import re
import copy
import pickle
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import nltk
from nltk import pos_tag

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import wordnet

with open(f'{os.path.abspath(os.path.dirname(__file__))}/assets/verb_tenses.pkl', 'rb') as handle:
    verb_tenses_dict = pickle.load(handle)


def _get_ohe_params(factual, has_ohe):
    """
    Find the group of columns related to a OHE encoding and their respective indexes
    :param factual: The factual point
    :param has_ohe: If True, it will try to get OHE information
    :return: (list) The first list has the groups of columns related to the OHE features while the second list has
    the index of all OHE encoded features
    """
    ohe_list = []
    ohe_indexes = []
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


def _seg_to_img(seg_arr, img, segments, replace_img):
    # Gets a segmentation code and transforms to image data

    converted_img = []
    for seg in seg_arr:
        mask_original = np.isin(segments, np.where(seg)[0]).astype(float)
        mask_replace = (mask_original == 0).astype(float)
        converted_img.append(
            img * np.stack((mask_original, mask_original, mask_original), axis=-1) +
            replace_img * np.stack((mask_replace, mask_replace, mask_replace), axis=-1))

    return converted_img


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


def _get_antonyms(word, pos):
    antonyms = []
    # If it's a negative word return positive
    negative_to_positive = {
        'not': '',
        "ain't": 'am',
        "aren't": 'are',
        "can't": 'can',
        "cannot": 'can',
        "can't've": 'can have',
        "couldn't": 'could',
        "couldn't've": 'could have',
        "didn't": 'did',
        "doesn't": 'does',
        "don't": 'do',
        "hadn't": 'had',
        "hadn't've": 'had have',
        "hasn't": 'has',
        "haven't": 'have',
        "isn't": 'is',
        "mayn't": 'may',
        "mightn't": 'might',
        "mightn't've": 'might have',
        "mustn't": 'must',
        "mustn't've": 'must have',
        "needn't": 'need',
        "needn't've": 'need have',
        "oughtn't": 'ought',
        "oughtn't've": 'ought have',
        "shan't": 'shall',
        "sha'n't": 'shall',
        "shan't've": 'shall have',
        "shouldn't": 'should',
        "shouldn't've": 'should have',
        "wasn't": 'was',
        "weren't": 'were',
        "won't": 'will',
        "won't've": 'will have',
        "wouldn't": 'would',
        "wouldn't've": 'would have'
    }
    if word in negative_to_positive.keys():
        return [word, negative_to_positive[word]]

    # Word types allowed for antonyms: JJ - Adjective, JJR - adjective comparative, JJS - Adjective superlative
    # RB - adverb, RBR - adverb comparative, RBS - adverb superlative
    # if pos in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
    for syn in wordnet.synsets(word):
        for syn_l in syn.lemmas():
            if syn_l.antonyms():
                antonym_word = syn_l.antonyms()[0].name()
                if antonym_word in verb_tenses_dict.keys():
                    if pos == 'VBD':
                        antonyms.append(verb_tenses_dict[antonym_word][1])
                    elif pos == 'VBG':
                        antonyms.append(verb_tenses_dict[antonym_word][3])
                    elif pos == 'VBN':
                        antonyms.append(verb_tenses_dict[antonym_word][2])
                    elif pos == 'VBZ':
                        antonyms.append(verb_tenses_dict[antonym_word][0])
                    else:
                        antonyms.append(antonym_word)
                else:
                    antonyms.append(antonym_word)

    antonyms = list(set(antonyms))
    if len(antonyms) > 0:
        antonyms = [word] + antonyms
    return antonyms


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
    text_replace_word = [[w, ''] if len(re.sub(r'\W+', '', w)) > 0 else [] for w in text_words]

    # This is a dictionary which keys are the original word position and values are the possible replacement
    change_vector_links = {idx: [*range(len(subs))] for idx, subs in enumerate(text_replace_word) if len(subs) > 0}

    # Now create the DataFrame representation
    # Create factual row values
    # Create column names
    factual_row = []
    factual_col = []
    for change_key, replace_idx in change_vector_links.items():
        factual_row.extend([1] + [0] * (len(replace_idx) - 1))
        factual_col.extend([f'{change_key}_{r}' for r in replace_idx])

    change_vector = pd.DataFrame([dict(zip(factual_col, factual_row))])

    return text_words, change_vector, text_replace_word


def _text_to_change_vector(text):
    # Verify and download NLTK packages
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    # First, make text lowercase
    text = text.lower()

    # Then get phrase tokens, replace ' with \\ to preserve contracted words
    text_words = nltk.word_tokenize(text.replace("'", "\\"))
    # Go back and replace \\ to '
    text_words = [w.replace("\\", "'") for w in text_words]

    # Get word part of speech
    text_words_pos = pos_tag(text_words)

    # Find antonyms for each word
    text_antonyms = [_get_antonyms(w, pos) for w, pos in text_words_pos]

    # If there's no antonyms, return warning
    if max([len(t) for t in text_antonyms]) == 0:
        warnings.warn('No word with antonym was found in the input text')

    # Now, for the replacement words, create a function that creates a vector, which initially represents the
    # factual phrase and can also represent replaced results.

    # This is a dictionary which keys are the original word position and values are the possible replacement
    change_vector_links = {idx: [*range(len(subs))] for idx, subs in enumerate(text_antonyms) if len(subs) > 0}
    # Now create the DataFrame representation
    # Create factual row values
    # Create column names
    factual_row = []
    factual_col = []
    for change_key, replace_idx in change_vector_links.items():
        factual_row.extend([1] + [0] * (len(replace_idx) - 1))
        factual_col.extend([f'{change_key}_{r}' for r in replace_idx])

    change_vector = pd.DataFrame([dict(zip(factual_col, factual_row))])

    return text_words, change_vector, text_antonyms


def _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html):
    n_rows = len(input_change_vector)
    out_texts = [copy.copy(text_words) for _ in range(n_rows)]

    # It can be a pandas DataFrame or numpy
    if type(input_change_vector) == pd.DataFrame:
        change_coordinates = list(input_change_vector.iloc[0][input_change_vector.iloc[0] == 1].index)
        change_coordinates = [[0] * len(change_coordinates), change_coordinates]
    else:
        change_coordinates = list(np.where(input_change_vector))
        change_coordinates[1] = [change_vector.columns[ci] for ci in change_coordinates[1]]

    for idx_cv in range(len(change_coordinates[0])):
        i_text, d_text = change_coordinates[0][idx_cv], change_coordinates[1][idx_cv]

        # Word index, Change index
        idx_w, idx_c = [int(x) for x in str(d_text).split('_')]

        # If change index is 0, do not make any alterations since it's the original word
        if idx_c != 0:
            word_replace = text_modifications[idx_w][idx_c]
            if not highlight_html:
                out_texts[i_text][idx_w] = word_replace
            else:
                # If the objective is to highlight changes, add bold and red
                if word_replace != '':
                    out_texts[i_text][idx_w] = f'<span style="color: red;font-weight: bold;">{word_replace}</span>'
                else:
                    out_texts[i_text][idx_w] = f'<span style="color: red;font-weight: bold; ' \
                                               f'text-decoration: line-through;">{out_texts[i_text][idx_w]}</span>'

    out_full_texts = [_untokenize(t) for t in out_texts]

    return out_full_texts


def _convert_change_vectors_func(text_words, change_vector, text_modifications):
    return lambda input_change_vector, highlight_html=False: _change_vector_to_text(
        input_change_vector, text_words, change_vector, text_modifications, highlight_html)
