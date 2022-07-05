"""
This module has the function that produces the CF using different types of cf finders and fine-tune.
"""
import copy
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import cv2

from ._cf_searchers import _random_generator, _super_sedc
from ._checkers import _check_factual, _check_vars, _check_prob_func
from ._data_standardizer import _get_ohe_params, _seg_to_img, _text_to_change_vector, _text_to_token_vector, \
    _convert_change_vectors_func
from ._fine_tune import _fine_tuning
from ._model_standardizer import _standardize_predictor, _adjust_model_class, _adjust_image_model, \
    _adjust_image_multiclass_nonspecific, _adjust_image_multiclass_second_best, _adjust_textual_classifier
from ._img_segmentation import gen_quickshift


class _CFBaseResponse:
    """
    Class that defines the base object to the CFNOW return
    """
    def __init__(self, factual, factual_vector, cf_vector, cf_not_optimized_vector, time_cf, time_cf_not_optimized):
        self.factual = factual
        self.factual_vector = factual_vector
        self.cf_vector = cf_vector
        self.cf_not_optimized_vector = cf_not_optimized_vector
        self.time_cf = time_cf
        self.time_cf_not_optimized = time_cf_not_optimized


class _CFTabular(_CFBaseResponse):
    """
    Class to return Tabular CF explanations
    """
    def __init__(self, **kwargs):

        super(_CFTabular, self).__init__(**kwargs)

        self.cf = self.cf_vector
        self.cf_not_optimized = self.cf_not_optimized_vector


class _CFImage(_CFBaseResponse):
    """
    Class to return Image CF explanations
    """
    def __init__(self, _seg_to_img, segments, replace_img, **kwargs):

        super(_CFImage, self).__init__(**kwargs)

        self.segments = segments
        self.cf = _seg_to_img([self.cf_vector], self.factual, segments, replace_img)[0]
        self.cf_not_optimized = _seg_to_img([self.cf_not_optimized_vector], self.factual, segments, replace_img)[0]

        self.cf_segments = np.where(self.cf_vector == 0)[0]
        self.cf_not_optimized_segments = np.where(self.cf_not_optimized_vector == 0)[0]

        # Green replacement to highlight changes
        green_replace = np.zeros(self.factual.shape)
        green_replace[:, :, 1] = 1
        self.cf_image_highlight = _seg_to_img([self.cf_vector], self.factual, segments, green_replace)[0]
        self.cf_not_optimized_image_highlight = _seg_to_img([self.cf_not_optimized_vector], self.factual,
                                                            segments, green_replace)[0]


class _CFText(_CFBaseResponse):
    """
    Class to return Text CF explanations
    """
    def __init__(self, converter, text_replace, **kwargs):

        super(_CFText, self).__init__(**kwargs)

        self.cf = converter([self.cf_vector])[0]
        self.cf_not_optimized = converter([self.cf_not_optimized_vector])[0]

        self.cf_html_highlight = converter([self.cf_vector], True)[0]
        self.cf_html_not_optimized = converter([self.cf_not_optimized_vector], True)[0]

        self.adjusted_factual = converter([self.factual_vector])[0]

        # Remove entries that are not considered (which does not have any word)
        text_replace_valid = np.array([t for t in text_replace if len(t) > 0])

        replaced_feats_idx = np.where(self.factual_vector != self.cf_vector)[0][::2]
        self.cf_replaced_words = [w[0] for w in text_replace_valid[replaced_feats_idx]]

        replaced_not_optimized_feats_idx = np.where(self.factual_vector != self.cf_not_optimized_vector)[0][::2]
        self.cf_not_optimized_replaced_words = [w[0] for w in text_replace_valid[replaced_not_optimized_feats_idx]]


def find_tabular(factual, model_predict_proba, feat_types=None, cf_strategy='greedy', increase_threshold=0, it_max=1000,
                 limit_seconds=30, ft_change_factor=0.1, ft_it_max=1000, size_tabu=5, ft_threshold_distance=0.01,
                 has_ohe=False, avoid_back_original=False, verbose=False):
    """
    For a factual tabular point and prediction model, finds a counterfactual explanation
    :param factual: The factual point as Pandas DataFrame
    :type factual: pandas.DataFrame
    :param model_predict_proba: Model's function which generates a class probability output
    :type model_predict_proba: object
    :param feat_types: Default: (all num). A dictionary with {col_name: col_type}, where "col_name" is the name of the
    column and "col_type" can be "num" to indicate numerical continuous features and "cat" to indicate categorical
    :type feat_types: dict
    :param cf_strategy: (optional) Strategy to find CF, can be "greedy" (default) or "random"
    :type cf_strategy: str
    :param increase_threshold: (optional) Threshold for improvement in CF score in the CF search,
    if the improvement is below that, search will stop. Default=0
    :type increase_threshold: int
    :param it_max: (optional) Maximum number of iterations for CF search. Default=1000
    :type it_max: int
    :param limit_seconds: (optional) Time threshold for CF optimization. Default=120
    :type limit_seconds: int
    :param ft_change_factor: (optional) Factor used for numerical features to change their values (e.g. if 0.1 it will
    use, initially, 10% of features' value). Default=0.1
    :type ft_change_factor: float
    :param ft_it_max: (optional) Maximum number of iterations for CF optimization step. Default=1000
    :type ft_it_max: int
    :param size_tabu: (optional) Size of Tabu Search list. Default=5
    :type size_tabu: int
    :param ft_threshold_distance: (optional) Threshold for CF optimization enhancement, if improvement is below the
    threshold, the optimization will be stopped. Default=0.01
    :type ft_threshold_distance: float
    :param has_ohe: (optional) True if you have one-hot encoded features. It will use the prefix (delimited by
    underscore char) to group different  features. For example, the columns: featName_1, featName_2 will
    be grouped as featName because they have the same prefix. Those features must be indicated in feat_types as "cat".
    Default=False
    :type has_ohe: bool
    :param avoid_back_original: For the greedy strategy, not allows changing back to the original values
    :type avoid_back_original: bool
    :param verbose: (optional) If True, it will output detailed information of CF finding and optimization steps.
    Default=False
    :type verbose: bool
    :return: (list) Containing [CF_array, CF_probability, CF_objective_value]
    """

    # Defines the type of data
    cf_data_type = 'tabular'

    # Defines the CF finding strategy
    cf_finder = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
    elif cf_strategy == 'greedy':
        cf_finder = _super_sedc
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

    # If feature types were not informed, all columns will be considered numerical
    if feat_types is None:
        feat_types = {c: 'num' for c in factual.index}

    # If Tabu size list is larger than the number of features issue a warning and reduce to size_features - 1
    if len(factual) < size_tabu:
        size_tabu_new = len(factual) - 1
        warnings.warn(f'The number of features ({len(factual)}) is lower than the Tabu list size ({size_tabu}),'
                      f'then, we reduced to the number of features minus 1 (={size_tabu_new})')
        size_tabu = size_tabu_new

    # Timer now
    time_start = datetime.now()

    # Make checks
    _check_factual(factual)
    _check_vars(factual, feat_types)
    _check_prob_func(factual, model_predict_proba)

    # Generate standardized predictor
    mp1 = _standardize_predictor(factual, model_predict_proba)

    # Correct class
    mp1c = _adjust_model_class(factual, mp1)

    # Generate OHE parameters if it has OHE variables
    ohe_list, ohe_indexes = _get_ohe_params(factual, has_ohe)
    # Generate CF using a CF finder
    cf_out = cf_finder(cf_data_type=cf_data_type,
                       factual=factual,
                       mp1c=mp1c,
                       feat_types=feat_types,
                       it_max=it_max,
                       ft_change_factor=ft_change_factor,
                       ohe_list=ohe_list,
                       ohe_indexes=ohe_indexes,
                       increase_threshold=increase_threshold,
                       tabu_list=None,
                       size_tabu=size_tabu,
                       avoid_back_original=avoid_back_original,
                       ft_time=None,
                       ft_time_limit=None,
                       verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    if mp1c(np.array([cf_out]))[0] < 0.5:
        raise Warning('No CF found')

    # Fine tune the counterfactual
    cf_out_ft = _fine_tuning(cf_data_type=cf_data_type,
                             factual=factual,
                             cf_out=cf_out,
                             mp1c=mp1c,
                             ohe_list=ohe_list,
                             ohe_indexes=ohe_indexes,
                             increase_threshold=increase_threshold,
                             feat_types=feat_types,
                             ft_change_factor=ft_change_factor,
                             it_max=it_max,
                             size_tabu=size_tabu,
                             ft_it_max=ft_it_max,
                             ft_threshold_distance=ft_threshold_distance,
                             time_start=time_start,
                             limit_seconds=limit_seconds,
                             cf_finder=cf_finder,
                             avoid_back_original=avoid_back_original,
                             verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    # Create response object
    response_obj = _CFTabular(
        factual=factual,
        factual_vector=factual,
        cf_vector=cf_out_ft[0],
        cf_not_optimized_vector=cf_out,
        time_cf=time_cf.total_seconds(),
        time_cf_not_optimized=time_cf_not_optimized.total_seconds())

    return response_obj


def find_image(img, model_predict, segmentation='quickshift', avoid_back_original=None, params_segmentation=None,\
               img_cf_strategy='nonspecific', cf_strategy='greedy', replace_mode='blur', increase_threshold=-1,
               it_max=None, limit_seconds=30, ft_change_factor=0.1, ft_it_max=None, size_tabu=None,
               ft_threshold_distance=0.01, has_ohe=False, verbose=False):
    """

    :param img: Image already processed to be classified, must be normalized between 0 and 1
    :param model_predict_proba:
    :param cf_strategy:
    :param increase_threshold:
    :param it_max:
    :param limit_seconds:
    :param ft_change_factor:
    :param ft_it_max:
    :param size_tabu:
    :param ft_threshold_distance:
    :param has_ohe:
    :param verbose:
    :return:
    """

    cf_data_type = 'image'

    cf_finder = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
        if it_max is None:
            it_max = 100
        if ft_it_max is None:
            ft_it_max = 100
        if avoid_back_original is None:
            avoid_back_original = False
    elif cf_strategy == 'greedy':
        cf_finder = _super_sedc
        if it_max is None:
            it_max = 1000
        if ft_it_max is None:
            ft_it_max = 1000
        if avoid_back_original is None:
            avoid_back_original = True
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

    # First, we need to create the segmentation for image
    if segmentation == 'quickshift':
        segments = gen_quickshift(img, params_segmentation)
    else:
        raise AttributeError(f'cf_strategy must be "quickshift" and not {segmentation}')

    # Then we create the image to be replaced (considered 0)
    if replace_mode == 'mean':
        replace_img = np.zeros(img.shape)
        replace_img[:,:,0], replace_img[:,:,1], replace_img[:,:,2] = img.mean(axis=(0,1))
    elif replace_mode == 'blur':
        replace_img = cv2.GaussianBlur(img, (31,31), 0)
    elif replace_mode == 'random':
        replace_img = np.random.random(img.shape)
    elif replace_mode == 'inpaint':
        replace_img = np.zeros(img.shape)
        for j in np.unique(segments):
            image_absolute = (img*255).astype('uint8')
            mask = np.full([image_absolute.shape[0], image_absolute.shape[1]], 0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            replace_img[segments == j] = image_segment_inpainted[segments == j]/255.0
    else:
        raise AttributeError(f'replace_mode must be "mean", "blur", "random" or "inpaint" and not {segmentation}')

    # Now create the factual, that depends of the segments and interpret them as binary features
    # Initially, all segments are activated (equal to 1)
    factual = pd.Series(np.array([1]*(np.max(segments) + 1)))

    # If not defined, tabu is the half the factual size
    if size_tabu is None:
        size_tabu = int(len(factual)/2)

    # If Tabu size list is larger than the number of segments issue a warning and reduce to size_features - 1
    if len(factual) < size_tabu:
        size_tabu_new = len(factual) - 1
        warnings.warn(f'The number of features ({len(factual)}) is lower than the Tabu list size ({size_tabu}),'
                      f'then, we reduced to the number of features minus 1 (={size_tabu_new})')
        size_tabu = size_tabu_new

    # Timer now
    time_start = datetime.now()

    # TODO: Make checks

    # Define all features as binary
    feat_types = {fidx: 'cat' for fidx in range(len(factual)) }

    # Then, the model must be modified to accept this new kind of data, where 1 means the original values and 0 the
    # modified values
    mic = _adjust_image_model(img, model_predict, segments, replace_img)

    # Get probability values adjusted
    if img_cf_strategy == 'nonspecific':
        mimns = _adjust_image_multiclass_nonspecific(factual, mic)
    elif img_cf_strategy == 'second_best':
        mimns = _adjust_image_multiclass_second_best(factual, mic)
    else:
        raise AttributeError(f'img_cf_strategy must be "nonspecific" or "second_best" and not {img_cf_strategy}')

    # Generate CF using a CF finder
    cf_out = cf_finder(cf_data_type=cf_data_type,
                       factual=factual,
                       mp1c=mimns,
                       feat_types=feat_types,
                       it_max=it_max,
                       ft_change_factor=ft_change_factor,
                       ohe_list=[],
                       ohe_indexes=[],
                       increase_threshold=increase_threshold,
                       tabu_list=None,
                       avoid_back_original=avoid_back_original,
                       size_tabu=size_tabu,
                       ft_time=None,
                       ft_time_limit=None,
                       verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    if mimns(np.array([cf_out]))[0] < 0.5:
        raise Warning('No CF found')

    # Fine tune the counterfactual
    cf_out_ft = _fine_tuning(cf_data_type=cf_data_type,
                             factual=factual,
                             cf_out=cf_out,
                             mp1c=mimns,
                             ohe_list=[],
                             ohe_indexes=[],
                             increase_threshold=increase_threshold,
                             feat_types=feat_types,
                             ft_change_factor=ft_change_factor,
                             it_max=it_max,
                             size_tabu=size_tabu,
                             ft_it_max=ft_it_max,
                             ft_threshold_distance=ft_threshold_distance,
                             time_start=time_start,
                             limit_seconds=limit_seconds,
                             cf_finder=cf_finder,
                             avoid_back_original=avoid_back_original,
                             verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    # Create response object
    response_obj = _CFImage(
        factual=img,
        factual_vector=factual,
        cf_vector=cf_out_ft[0],
        cf_not_optimized_vector=cf_out,
        time_cf=time_cf.total_seconds(),
        time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

        _seg_to_img=_seg_to_img,
        segments=segments,
        replace_img=replace_img
    )

    return response_obj


def find_text(text_input, textual_classifier, cf_strategy='greedy', word_replace_strategy='remove',
              increase_threshold=-1, it_max=1000, limit_seconds=30, ft_change_factor=0.1, ft_it_max=1000,
              size_tabu=None, ft_threshold_distance=0.01, has_ohe=False, avoid_back_original=False, verbose=False):
    # Textual predictor must receive an array of texts and output a class between 0 and 1

    cf_data_type = 'text'

    cf_finder = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
    elif cf_strategy == 'greedy':
        cf_finder = _super_sedc
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

    # Timer now
    time_start = datetime.now()

    # TODO: Make checkers

    # Define type of word replacement strategy
    if word_replace_strategy == 'remove':
        text_words, change_vector, text_replace = _text_to_token_vector(text_input)
    elif word_replace_strategy == 'antonyms':
        text_words, change_vector, text_replace = _text_to_change_vector(text_input)
    else:
        raise AttributeError(f'word_replace_strategy must be "antonyms" and not {word_replace_strategy}')

    converter = _convert_change_vectors_func(text_words, change_vector, text_replace)
    factual = copy.copy(change_vector)

    feat_types = {c: 'cat' for c in factual.columns}

    original_text_classification = np.array(textual_classifier([text_input])).reshape(-1)[0]

    mt = _adjust_textual_classifier(textual_classifier, converter, original_text_classification)

    # Standardize the predictor output
    mts = _standardize_predictor(factual.iloc[0], mt)

    encoded_text_classification = mts(factual)[0]

    # Verify if the encoded classification is equal to the original classification
    # the first part must be adjusted if the score is higher than 1 since the model will flip the class
    if (original_text_classification if original_text_classification < 0.5 else 1 - original_text_classification) != encoded_text_classification:
        print('The original text and classifier have a different result compared with the encoded text and '
              'adapted model. PLEASE REPORT THIS BUG, PREFERABLY WITH THE DATA AND MODEL USED.')

    # Generate OHE parameters if it has OHE variables
    ohe_list, ohe_indexes = _get_ohe_params(factual.iloc[0], True)

    # Define Tabu list size if it's none
    if size_tabu is None:
        size_tabu = int(len(ohe_list)/2)

    # Generate CF using a CF finder
    cf_out = cf_finder(cf_data_type=cf_data_type,
                       factual=factual.iloc[0],
                       mp1c=mts,
                       feat_types=feat_types,
                       it_max=it_max,
                       ft_change_factor=ft_change_factor,
                       ohe_list=ohe_list,
                       ohe_indexes=ohe_indexes,
                       increase_threshold=increase_threshold,
                       tabu_list=None,
                       size_tabu=size_tabu,
                       avoid_back_original=avoid_back_original,
                       ft_time=None,
                       ft_time_limit=None,
                       verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    # If no CF was found, return original text, since this may be common, it will not raise errors
    if mts(np.array([cf_out]))[0] < 0.5:
        print('No CF found')
        return text_input, text_input, factual, factual

    # Fine tune the counterfactual
    cf_out_ft = _fine_tuning(cf_data_type=cf_data_type,
                             factual=factual.iloc[0],
                             cf_out=cf_out,
                             mp1c=mts,
                             ohe_list=ohe_list,
                             ohe_indexes=ohe_indexes,
                             increase_threshold=increase_threshold,
                             feat_types=feat_types,
                             ft_change_factor=ft_change_factor,
                             it_max=it_max,
                             size_tabu=size_tabu,
                             ft_it_max=ft_it_max,
                             ft_threshold_distance=ft_threshold_distance,
                             time_start=time_start,
                             limit_seconds=limit_seconds,
                             cf_finder=cf_finder,
                             avoid_back_original=avoid_back_original,
                             verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    # Create response object
    response_obj = _CFText(
        factual=text_input,
        factual_vector=factual.to_numpy()[0],
        cf_vector=cf_out_ft[0],
        cf_not_optimized_vector=cf_out,
        time_cf=time_cf.total_seconds(),
        time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

        converter=converter,
        text_replace=text_replace,
    )

    return response_obj
