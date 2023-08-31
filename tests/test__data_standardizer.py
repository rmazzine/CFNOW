import unittest
from unittest.mock import patch, MagicMock, call

import pandas as pd
import numpy as np
import nltk

from cfnow._data_standardizer import _get_ohe_params, _ohe_detector, _get_ohe_list, _seg_to_img, _untokenize, \
    _get_antonyms, _text_to_token_vector, _text_to_change_vector, _change_vector_to_text, _convert_change_vectors_func

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

class TestScriptBase(unittest.TestCase):

    # Simple image, segments and replace_img to be used in tests
    img = []
    segments = []
    replace_img = []
    for ir in range(240):
        row_img = []
        row_segments = []
        row_replace_image = []
        for ic in range(240):
            row_replace_image.append([0, 0, 0])
            if ir > ic:
                row_img.append([127, 127, 127])
                row_segments.append(1)
            else:
                row_img.append([255, 255, 255])
                row_segments.append(0)
        replace_img.append(row_replace_image)
        img.append(row_img)
        segments.append(row_segments)
    replace_img = np.array(replace_img).astype('uint8')
    img = np.array(img).astype('uint8')
    segments = np.array(segments)

    def test__get_ohe_params_no_ohe(self):
        factual = pd.Series({'0': 1, '1': 0, '2_0': 1, '2_1': 0, '2_2': 0})
        has_ohe = False

        ohe_list, ohe_indexes = _get_ohe_params(factual, has_ohe)

        self.assertListEqual(ohe_list, [])

        self.assertListEqual(ohe_indexes, [])

    def test__get_ohe_params_ohe(self):
        factual = pd.Series({'0': 1, '1': 0, '2_0': 1, '2_1': 0, '2_2': 0})
        has_ohe = True

        ohe_list, ohe_indexes = _get_ohe_params(factual, has_ohe)

        self.assertListEqual(ohe_list, [[2, 3, 4]])

        self.assertListEqual(ohe_indexes, [2, 3, 4])

    def test__get_ohe_params_ohe_but_all_num_bin(self):
        factual = pd.Series({'0': 1, '1': 0})
        has_ohe = True

        ohe_list, ohe_indexes = _get_ohe_params(factual, has_ohe)

        self.assertListEqual(ohe_list, [])

        self.assertListEqual(ohe_indexes, [])

    def test__ohe_detector_no_match(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 5, 6]

        result_ohe_detector = _ohe_detector(lst1, lst2)

        self.assertFalse(result_ohe_detector)

    def test__ohe_detector_single_match(self):
        lst1 = [1, 2, 3]
        lst2 = [3, 4, 5]

        result_ohe_detector = _ohe_detector(lst1, lst2)

        self.assertFalse(result_ohe_detector)

    def test__ohe_detector_match(self):
        lst1 = [1, 3, 4]
        lst2 = [3, 4, 5]

        result_ohe_detector = _ohe_detector(lst1, lst2)

        self.assertTrue(result_ohe_detector)

    def test__get_ohe_list_not_in(self):
        f_idx = 0
        ohe_list = [[1, 2, 3]]

        result_get_ohe_list = _get_ohe_list(f_idx, ohe_list)

        self.assertIsNone(result_get_ohe_list)

    def test__get_ohe_list_in(self):
        f_idx = 1
        ohe_list = [[1, 2, 3]]

        result_get_ohe_list = _get_ohe_list(f_idx, ohe_list)

        self.assertListEqual(result_get_ohe_list, [1, 2, 3])

    def test__seg_to_img_original_image(self):
        # There are two segments (0 and 1), both are 1 representing they belong to the original image
        seg_arr = [pd.Series({0: 1, 1: 1})]
        img = self.img
        segments = self.segments
        replace_img = self.replace_img

        response_seg_to_img = _seg_to_img(seg_arr, img, segments, replace_img)

        # Verify size of response (since we only send one segment array we must receive only one image)
        self.assertEqual(len(response_seg_to_img), 1)
        # Since all segments are original the sum of different arrays must be zero
        self.assertEqual((img != response_seg_to_img[0]).sum(), 0)

    def test__seg_to_img_replace_image(self):
        # There are two segments (0 and 1), both are 0 representing they belong to the modified image
        seg_arr = [pd.Series({0: 0, 1: 0})]
        img = self.img
        segments = self.segments
        replace_img = self.replace_img

        response_seg_to_img = _seg_to_img(seg_arr, img, segments, replace_img)

        # Verify size of response (since we only send one segment array we must receive only one image)
        self.assertEqual(len(response_seg_to_img), 1)
        # Since all segments are the modified, no pixel should be equal
        self.assertEqual((img == response_seg_to_img[0]).sum(), 0)

    def test__seg_to_img_replace_half_image(self):
        # There are two segments (0 and 1), the first (0) is the modified and the second (1) is the original
        seg_arr = [pd.Series({0: 0, 1: 1})]
        img = self.img
        segments = self.segments
        replace_img = self.replace_img

        response_seg_to_img = _seg_to_img(seg_arr, img, segments, replace_img)

        # Verify size of response (since we only send one segment array we must receive only one image)
        self.assertEqual(len(response_seg_to_img), 1)
        # Since all segments are the modified, no pixel should be equal
        self.assertEqual((img == response_seg_to_img[0]).sum(), (segments == 1).sum()*3)

    def test__untokenize_sample(self):
        words = ['did', 'you', 'see', 'my', 'cat', '?', 'He', 'is', 'beautiful', '.', '.', '.', 'he', 'likes', 'loves',
                 "'", 'cat', "'", 'food', 'like', 'lasagna', '!', 'He', 'is', "n't", 'a', 'common', 'cat', 'and',
                 'can', 'not', 'be', 'outside', '.']

        result_untokenize = _untokenize(words)

        self.assertEqual(result_untokenize, "did you see my cat? He is beautiful... he likes loves' cat' food like "
                                            "lasagna! He isn't a common cat and cannot be outside.")

    def test__untokenize_nothing(self):
        words = []

        result_untokenize = _untokenize(words)

        self.assertEqual(result_untokenize, "")

    def test__get_antonyms_negative(self):
        # Some examples
        words = ['not', "can't", "haven't", "wasn't"]
        poss = ['', '', '', '']

        return_get_antonyms = []
        for w, p in zip(words, poss):
            return_get_antonyms.append(_get_antonyms(w, p))

        self.assertListEqual(
            [['not', ''], ["can't", 'can'], ["haven't", 'have'], ["wasn't", 'was']], return_get_antonyms)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_without_antonym(self, mock_wordnet):
        # Some examples
        word = 'house'
        pos = ''

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = None

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()

        self.assertListEqual([], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_not_verb(self, mock_wordnet):
        # Some examples
        word = 'good'
        pos = ''

        mock_name = MagicMock()
        mock_name.name.return_value = 'bad'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['good', 'bad'], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_verb_VBD(self, mock_wordnet):
        # Some examples
        word = 'loved'
        pos = 'VBD'

        mock_name = MagicMock()
        mock_name.name.return_value = 'hate'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['loved', 'hated'], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_verb_VBG(self, mock_wordnet):
        # Some examples
        word = 'loving'
        pos = 'VBG'

        mock_name = MagicMock()
        mock_name.name.return_value = 'hate'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['loving', 'hating'], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_verb_VBN(self, mock_wordnet):
        # Some examples
        word = 'loved'
        pos = 'VBN'

        mock_name = MagicMock()
        mock_name.name.return_value = 'hate'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['loved', 'hated'], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_verb_VBZ(self, mock_wordnet):
        # Some examples
        word = 'loves'
        pos = 'VBZ'

        mock_name = MagicMock()
        mock_name.name.return_value = 'hate'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['loves', 'hates'], return_get_antonym)

    @patch('cfnow._data_standardizer.wordnet')
    def test__get_antonyms_antonym_verb_OTHER(self, mock_wordnet):
        # Some examples
        word = 'love'
        pos = 'OTHER'

        mock_name = MagicMock()
        mock_name.name.return_value = 'hate'

        mock_lemmas = MagicMock()
        mock_lemmas.antonyms.return_value = [mock_name]

        mock_syn = MagicMock()
        mock_syn.lemmas.return_value = [mock_lemmas]

        mock_wordnet.synsets.return_value = [mock_syn]

        return_get_antonym = _get_antonyms(word, pos)

        mock_wordnet.synsets.assert_called_once()
        mock_syn.lemmas.assert_called_once()
        mock_lemmas.antonyms.assert_called()
        mock_name.name.assert_called_once()

        self.assertListEqual(['love', 'hate'], return_get_antonym)

    @patch('cfnow._data_standardizer.nltk')
    def test__text_to_token_vector_download_punkt(self, mock_nltk):

        mock_nltk.data.find.side_effect = LookupError()

        text = 'This is an example text'

        result_text_to_token_vector = _text_to_token_vector(text)

        mock_nltk.data.find.assert_called_once_with('tokenizers/punkt')
        mock_nltk.download.assert_called_once_with('punkt')

    def test__text_to_token_vector_tokenize_example(self):

        text = "This, a test for tokenization, ensure codes' quality that's fundamental for science. Fora Bolsonaro."

        text_words_expected = ['This', ',', 'a', 'test', 'for', 'tokenization', ',', 'ensure', "codes'", 'quality',
                               "that's", 'fundamental', 'for', 'science', '.', 'Fora', 'Bolsonaro', '.']

        text_replace_word_expected = [['This', ''], [], ['a', ''], ['test', ''], ['for', ''], ['tokenization', ''], [],
                                      ['ensure', ''], ["codes'", ''], ['quality', ''], ["that's", ''],
                                      ['fundamental', ''], ['for', ''], ['science', ''], [], ['Fora', ''],
                                      ['Bolsonaro', ''], []]

        # The expected dict must have n_0 and n_1 for each word, only words that have a replacement ,the length
        # of the replace_word is larger than zero, are considered.
        dict_change_vect_values = {}
        for idx_we, we in enumerate(text_replace_word_expected):
            if len(we) > 0:
                dict_change_vect_values[f'{idx_we}_0'] = 1
                dict_change_vect_values[f'{idx_we}_1'] = 0
        change_vector_expected = pd.DataFrame([dict_change_vect_values])

        text_words, change_vector, text_replace_word = _text_to_token_vector(text)

        self.assertListEqual(text_words, text_words_expected)
        self.assertListEqual(list(change_vector.columns), list(change_vector_expected.columns))
        self.assertListEqual(change_vector.values.tolist(), change_vector_expected.values.tolist())
        self.assertListEqual(text_replace_word, text_replace_word_expected)

    def test__text_to_change_vector_example(self):

        text = "The medieval age is great! It has many great stories to tell. You cannot miss that, you will love it!"

        text_words_expected = ['the', 'medieval', 'age', 'is', 'great', '!', 'it', 'has', 'many', 'great',
                               'stories', 'to', 'tell', '.', 'you', 'can', 'not', 'miss', 'that', ',', 'you',
                               'will', 'love', 'it', '!']

        text_antonyms_expected = [[], [], ['age', 'rejuvenate'], ['is', 'differs'], [], [], [],
                                  ['has', 'refuses', 'abstains', 'lacks'], ['many', 'few'], [], [], [], [], [], [],
                                  ['can', 'hire'], ['not', ''], ['miss', 'hit', 'attend', 'attend_to', 'have'],
                                  [], [], [], ['will', 'disinherit'], ['love', 'hate'], [], []]

        # The expected dict must have n_0, n_1, ..., n_z  (z is the number of antonyms) for each word,
        # only words that have a replacement ,the length of the replace_word is larger than zero, are considered.
        dict_change_vect_values = {}
        for idx_we, we in enumerate(text_antonyms_expected):
            if len(we) > 0:
                dict_change_vect_values[f'{idx_we}_0'] = 1
                for z in range(len(we) - 1):
                    dict_change_vect_values[f'{idx_we}_{z + 1}'] = 0
        change_vector_expected = pd.DataFrame([dict_change_vect_values])

        text_words, change_vector, text_antonyms = _text_to_change_vector(text)

        self.assertListEqual(text_words, text_words_expected)
        self.assertListEqual(list(change_vector.columns), list(change_vector_expected.columns))
        self.assertListEqual(change_vector.values.tolist(), change_vector_expected.values.tolist())
        self.assertListEqual([set(e) for e in text_antonyms], [set(e) for e in text_antonyms_expected])

    @patch('cfnow._data_standardizer.warnings')
    def test__text_to_change_vector_empty(self, mock_warnings):

        text = "test"

        text_words, change_vector, text_antonyms = _text_to_change_vector(text)

        mock_warnings.warn.assert_called_once()

        self.assertListEqual(text_words, ['test'])
        self.assertListEqual(list(change_vector.columns), [])
        self.assertListEqual(change_vector.values.tolist(), [[]])
        self.assertListEqual(text_antonyms, [[]])

    @patch('cfnow._data_standardizer._get_antonyms')
    @patch('cfnow._data_standardizer.pos_tag')
    @patch('cfnow._data_standardizer.nltk')
    def test__text_to_change_vector_test_download_nltk_packages(self, mock_nltk, mock_pos_tag, mock__get_antonyms):

        text = 'I like the tests I made here.'

        mock_nltk.data.find.side_effect = LookupError()

        mock_nltk.word_tokenize.return_value = ['i', 'like', 'the', 'tests', 'i', 'made', 'here', '.']
        mock_pos_tag.return_value = [('i', 'NN'), ('like', 'IN'), ('the', 'DT'), ('tests', 'NNS'),
                                     ('i', 'VBP'), ('made', 'VBN'), ('here', 'RB'), ('.', '.')]
        mock__get_antonyms.side_effect = [[], ['like', 'dislike', 'unlike', 'unalike'], [], [], [],
                                          ['made', 'unmade', 'broken'], ['here', 'there'], []]

        text_words, change_vector, text_antonyms = _text_to_change_vector(text)

        self.assertListEqual(mock_nltk.download.mock_calls, [call('punkt'), call('wordnet'),
                                                             call('omw-1.4'), call('averaged_perceptron_tagger')])

    @patch('cfnow._data_standardizer._untokenize')
    def test__change_vector_to_text_example_pd_dataframe(self, mock_untokenize):

        input_change_vector = pd.DataFrame([{'0_0': 0, '0_1': 1, '1_0': 1, '1_1': 0, '2_0': 0, '2_1': 1}])
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        text_modifications = [['I', ''], ['like', ''], ['music', '']]
        highlight_html = False

        mock_untokenize.return_value = 'like'

        _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html)

        mock_untokenize.assert_called_once_with(['', 'like', ''])

    @patch('cfnow._data_standardizer._untokenize')
    def test__change_vector_to_text_example_np_array(self, mock_untokenize):
        input_change_vector = [np.array([0, 1, 0, 1, 1, 0])]
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        text_modifications = [['I', ''], ['like', ''], ['music', '']]
        highlight_html = False

        mock_untokenize.return_value = 'music'

        _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html)

        mock_untokenize.assert_called_once_with(['', '', 'music'])

    @patch('cfnow._data_standardizer._untokenize')
    def test__change_vector_to_text_example_multiple_replace_options_np_array(self, mock_untokenize):
        input_change_vector = [np.array([0, 1, 0, 0])]
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'1_0': 1, '1_1': 0, '1_2': 0, '1_3': 0}])
        text_modifications = [[], ['like', 'unlike', 'unalike', 'dislike'], []]
        highlight_html = False

        mock_untokenize.return_value = 'I unlike music'

        _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html)

        mock_untokenize.assert_called_once_with(['I', 'unlike', 'music'])

    @patch('cfnow._data_standardizer._untokenize')
    def test__change_vector_to_text_example_np_array_highlight_html(self, mock_untokenize):
        input_change_vector = [np.array([0, 1, 0, 1, 1, 0])]
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        text_modifications = [['I', ''], ['like', ''], ['music', '']]
        highlight_html = True

        mock_untokenize.return_value = \
            '<span style="color: red;font-weight: bold; text-decoration: line-through;">I</span> ' \
            '<span style="color: red;font-weight: bold; text-decoration: line-through;">like</span> music'

        _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html)

        mock_untokenize.assert_called_once_with([
            '<span style="color: red;font-weight: bold; text-decoration: line-through;">I</span>',
            '<span style="color: red;font-weight: bold; text-decoration: line-through;">like</span>',
            'music'])

    @patch('cfnow._data_standardizer._untokenize')
    def test__change_vector_to_text_example_multiple_replace_options_np_array_highlight_html(self, mock_untokenize):
        input_change_vector = [np.array([0, 1, 0, 0])]
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'1_0': 1, '1_1': 0, '1_2': 0, '1_3': 0}])
        text_modifications = [[], ['like', 'unlike', 'unalike', 'dislike'], []]
        highlight_html = True

        mock_untokenize.return_value = 'I <span style="color: red;font-weight: bold;">unlike</span> music'

        _change_vector_to_text(input_change_vector, text_words, change_vector, text_modifications, highlight_html)

        mock_untokenize.assert_called_once_with([
            'I', '<span style="color: red;font-weight: bold;">unlike</span>', 'music'])

    @patch('cfnow._data_standardizer._change_vector_to_text')
    def test__convert_change_vectors_func_example(self, mock__change_vector_to_text):
        text_words = ['I', 'like', 'music']
        change_vector = pd.DataFrame([{'0_0': 1, '0_1': 0, '1_0': 1, '1_1': 0, '2_0': 1, '2_1': 0}])
        text_modifications = [['I', ''], ['like', ''], ['music', '']]

        input_change_vector = [np.array([0, 1, 1, 0, 1, 0])]
        highlight_html = False

        function_convert_change_vectors_func = _convert_change_vectors_func(
            text_words, change_vector, text_modifications)

        function_convert_change_vectors_func(input_change_vector, highlight_html)

        mock__change_vector_to_text.assert_called_once()
