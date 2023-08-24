"""
This module has the function that produces the CF using different types of cf finders and fine-tune.
"""
import copy
import logging
import warnings
from datetime import datetime
from typing import Literal

import counterplots as cpl
import pandas as pd
import numpy as np
import cv2

from ._cf_searchers import _random_generator, _greedy_generator
from ._checkers import _check_factual, _check_vars, _check_prob_func
from ._data_standardizer import _get_ohe_params, _seg_to_img, _text_to_change_vector, _text_to_token_vector, \
    _convert_change_vectors_func
from ._fine_tune import _fine_tuning
from ._model_standardizer import _standardize_predictor, _adjust_model_class, _adjust_image_model, \
    _adjust_multiclass_nonspecific, _adjust_multiclass_second_best, _adjust_textual_classifier
from ._img_segmentation import gen_quickshift


class _CFBaseResponse:
    """
    Class that defines the base object to the CFNOW return
    """
    def __init__(self, factual, factual_vector, cf_vectors, cf_not_optimized_vectors, obj_scores,
                 time_cf, time_cf_not_optimized, model_pred=None):
        """
        This receives the base parameters that all CF should have
        :param factual: The factual instance provided by the user
        :param factual_vector: The vector representation of the factual instance
        :param cf_vectors: Vectors representation of the CFs
        :param cf_not_optimized_vectors: Vectors representation (not optimized) of the CFs
        :param obj_scores: The objective scores of the CFs
        :param time_cf: Time spent to generate the optimized CF
        :param time_cf_not_optimized: Time spent to generate the not optimized CF
        """
        self.factual = factual
        self.factual_vector = factual_vector
        self.cf_vectors = cf_vectors
        self.cf_not_optimized_vectors = cf_not_optimized_vectors
        self.time_cf = time_cf
        self.time_cf_not_optimized = time_cf_not_optimized
        self.total_cf = len(cf_vectors)
        self.cf_obj_scores = obj_scores
        self.model_pred = model_pred


class _CFTabular(_CFBaseResponse):
    """
    Class to return Tabular CF explanations
    """
    def __init__(self, **kwargs):

        super(_CFTabular, self).__init__(**kwargs)

        self.cfs = self.cf_vectors
        self.cfs_not_optimized = self.cf_not_optimized_vectors

    def generate_counterplots(self, idx_cf):
        """
        Generates the counterplots for a given CF
        :param idx_cf: The index of the CF to generate the counterplots
        :return: A dictionary with the counterplots
        """
        return cpl.CreatePlot(
            factual=np.array(self.factual.tolist()),
            cf=np.array(self.cfs[idx_cf].tolist()),
            model_pred=lambda x: self.model_pred(x)[:, 0])


class _CFImage(_CFBaseResponse):
    """
    Class to return Image CF explanations
    """
    def __init__(self, _seg_to_img, segments, replace_img, **kwargs):
        """
        Image CF
        :param _seg_to_img: Function which transforms a segment vector to an image
        :param segments: The map of segments of the image
        :param replace_img: The image to be replaced
        """

        super(_CFImage, self).__init__(**kwargs)

        self.segments = segments

        # Green replacement to highlight changes
        green_replace = np.zeros(self.factual.shape)
        green_replace[:, :, 1] = 1

        cfs = []
        cfs_segments = []
        cfs_image_highlight = []

        cfs_not_optimized = []
        cfs_not_optimized_segments = []
        cfs_not_optimized_image_highlight = []

        for cf_vector in self.cf_vectors:
            cfs.append(_seg_to_img([cf_vector], self.factual, segments, replace_img)[0])
            cfs_segments.append(np.where(cf_vector == 0)[0])
            cfs_image_highlight.append(_seg_to_img([cf_vector], self.factual, segments, green_replace)[0])

        for cf_not_optimized_vector in self.cf_not_optimized_vectors:
            cfs_not_optimized.append(_seg_to_img([cf_not_optimized_vector], self.factual, segments, replace_img)[0])
            cfs_not_optimized_segments.append(np.where(cf_not_optimized_vector == 0)[0])
            cfs_not_optimized_image_highlight.append(
                _seg_to_img([cf_not_optimized_vector], self.factual, segments, green_replace)[0])

        self.cfs = cfs
        self.cfs_segments = cfs_segments
        self.cfs_image_highlight = cfs_image_highlight
        self.cfs_not_optimized = cfs_not_optimized
        self.cfs_not_optimized_segments = cfs_not_optimized_segments
        self.cfs_not_optimized_image_highlight = cfs_not_optimized_image_highlight


class _CFText(_CFBaseResponse):
    """
    Class to return Text CF explanations
    """
    def __init__(self, converter, text_replace, **kwargs):
        """
        Text CF
        :param converter: Function to convert text vector representations to text
        :param text_replace: List with words from text and the options to be replaced
        """

        super(_CFText, self).__init__(**kwargs)

        # Remove entries that are not considered (which does not have any word)
        text_replace_valid = np.array([t for t in text_replace if len(t) > 0])

        self.adjusted_factual = converter([self.factual_vector])[0]

        cfs = []
        cfs_html_highlight = []
        cfs_replaced_words = []

        cfs_not_optimized = []
        cfs_not_optimized_html_highlight = []
        cfs_not_optimized_replaced_words = []

        for cf_vector in self.cf_vectors:
            cfs.append(converter([cf_vector])[0])
            cfs_html_highlight.append(converter([cf_vector], True)[0])
            replaced_feats_idx = [int(wi / 2) for wi in np.where(self.factual_vector != cf_vector)[0][::2]]
            cfs_replaced_words.append([w[0] for w in text_replace_valid[replaced_feats_idx]])

        for cf_not_optimized_vector in self.cf_not_optimized_vectors:
            cfs_not_optimized.append(converter([cf_not_optimized_vector])[0])
            cfs_not_optimized_html_highlight.append(converter([cf_not_optimized_vector], True)[0])
            replaced_not_optimized_feats_idx = [int(wi / 2) for wi in
                                                np.where(self.factual_vector != cf_not_optimized_vector)[0][::2]]
            cfs_not_optimized_replaced_words.append(
                [w[0] for w in text_replace_valid[replaced_not_optimized_feats_idx]])

        self.cfs = cfs
        self.cfs_html_highlight = cfs_html_highlight
        self.cfs_replaced_words = cfs_replaced_words
        self.cfs_not_optimized = cfs_not_optimized
        self.cfs_not_optimized_html_highlight = cfs_not_optimized_html_highlight
        self.cfs_not_optimized_replaced_words = cfs_not_optimized_replaced_words


def _define_tabu_size(size_tabu, factual_vector):
    """
    Define the tabu size for the tabu search
    :param size_tabu: The tabu size provided by the user
    :param factual_vector: Factual vector which a CF will be created
    :return: The tabu size to be used
    """
    if type(size_tabu) is float:
        if size_tabu < 0.0 or size_tabu > 1.0:
            raise AttributeError(f'size_tabu must be between 0.0 and 1.0 and not {size_tabu}')
        size_tabu = int(round(size_tabu * len(factual_vector.index)))
    else:
        # If Tabu size list is larger than the number of features issue a warning and reduce to size_features - 1
        if len(factual_vector) < size_tabu:
            size_tabu_new = len(factual_vector) - 1
            warnings.warn(f'The number of features ({len(factual_vector)}) is lower than the Tabu list size ({size_tabu}),'
                          f'then, we reduced to the number of features minus 1 (={size_tabu_new})')
            size_tabu = size_tabu_new
    return size_tabu


def find_tabular(
        factual: pd.Series,
        model_predict_proba,
        count_cf: int = 1,
        feat_types: dict = None,
        cf_strategy: Literal['greedy', 'random', 'random-sequential'] = 'random',
        increase_threshold: int = 0,
        it_max: int = 5000,
        limit_seconds: int = 120,
        ft_change_factor: float = 0.1,
        ft_it_max: int = None,
        size_tabu: (int, float) = None,
        ft_threshold_distance: float = 1e-05,
        has_ohe: bool = False,
        avoid_back_original: bool = False,
        threshold_changes: int = 100,
        verbose: bool = False) -> _CFTabular:
    """
    For a factual tabular point and prediction model, finds a counterfactual explanation
    :param factual: The factual point as Pandas DataFrame
    :type factual: pandas.DataFrame
    :param model_predict_proba: Model's function which generates a class probability output, it must be able to accept
    a Pandas DataFrame and Numpy array as input
    :type model_predict_proba: object
    :param count_cf: Number of counterfactual explanations to be returned
    :type count_cf: int
    :param feat_types: (optional) A dictionary with {col_name: col_type}, where "col_name" is the name of the column
    and "col_type" can be "num" to indicate numerical continuous features and "cat" to indicate
    categorical Default: (all num)
    :type feat_types: dict
    :param cf_strategy: (optional) Strategy to find CF, can be "greedy", "random" or
    "random-sequential". Default='random'
    :type cf_strategy: str
    :param increase_threshold: (optional) Threshold for improvement in CF score in the CF search,
    if the improvement is below that, search will stop. -1 deactivates this check. Default=1e-05
    :type increase_threshold: int, float
    :param it_max: (optional) Maximum number of iterations for CF search. Default=5000
    :type it_max: int
    :param limit_seconds: (optional) Time threshold for CF optimization. Default=120
    :type limit_seconds: int
    :param ft_change_factor: (optional) Factor used for numerical features to change their values (e.g. if 0.1 it will
    use, initially, 10% of features' value). Default=0.1
    :type ft_change_factor: float
    :param ft_it_max: (optional) Maximum number of iterations for CF optimization step. Default=1000 (greedy)
    or 5000 (random)
    :type ft_it_max: int
    :param size_tabu: (optional) Size of tabu list, if float it's the share of features,
    if int it's the exact number defined. Default=5 (greedy) or 0.2 (random)
    :type size_tabu: int, float
    :param ft_threshold_distance: (optional) Threshold for CF optimization enhancement, if improvement is below the
    threshold, the optimization will be stopped. -1 deactivates this check. Default=1e-05
    :type ft_threshold_distance: float
    :param has_ohe: (optional) True if you have one-hot encoded features. It will use the prefix (delimited by
    underscore char) to group different  features. For example, the columns: featName_1, featName_2 will
    be grouped as featName because they have the same prefix. Those features must be indicated in feat_types as "cat".
    Default=False
    :type has_ohe: bool
    :param avoid_back_original: (optional) For the greedy strategy, not allows changing back to the original values.
    Default=False
    :type avoid_back_original: bool
    :param threshold_changes: (optional) For the random strategy, threshold for the maximum number of changes to be
    created in the CF finding process. Default=100
    :type threshold_changes: int
    :param verbose: (optional) If True, it will output detailed information of CF finding and optimization steps.
    Default=False
    :type verbose: bool
    :return: Object with CF information
    :rtype: _CFTabular
    """

    # Defines the type of data
    cf_data_type = 'tabular'

    # Defines the CF finding strategy
    cf_finder = None
    finder_strategy = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
        if ft_it_max is None:
            ft_it_max = 5000
        if size_tabu is None:
            size_tabu = 0.2
    elif cf_strategy == 'random-sequential':
        cf_finder = _random_generator
        finder_strategy = 'sequential'
        if ft_it_max is None:
            ft_it_max = 5000
        if size_tabu is None:
            size_tabu = 0.2
    elif cf_strategy == 'greedy':
        cf_finder = _greedy_generator
        if ft_it_max is None:
            ft_it_max = 1000
        if size_tabu is None:
            size_tabu = 5
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "greedy", "random" or "random-sequential" and not {cf_strategy}')

    # If feature types were not informed, all columns will be considered numerical
    if feat_types is None:
        feat_types = {c: 'num' for c in factual.index}

    size_tabu = _define_tabu_size(size_tabu, factual)

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

    # Timer not optimized CF
    time_start = datetime.now()

    # Generate CF using a CF finder
    cf_unique = cf_finder(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
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
        threshold_changes=threshold_changes,
        count_cf=count_cf,
        cf_unique=[],
        verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    if len(cf_unique) == 0:
        logging.log(30, 'No CF found.')
        return _CFTabular(
            factual=factual,
            factual_vector=factual,
            cf_vectors=[],
            cf_not_optimized_vectors=[],
            obj_scores=[],
            time_cf=time_cf_not_optimized.total_seconds(),
            time_cf_not_optimized=time_cf_not_optimized.total_seconds())

    # Fine tune the counterfactual
    cf_unique_opt = _fine_tuning(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
        factual=factual,
        cf_unique=cf_unique,
        count_cf=count_cf,
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
        limit_seconds=limit_seconds,
        cf_finder=cf_finder,
        avoid_back_original=avoid_back_original,
        threshold_changes=threshold_changes,
        verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    return _CFTabular(
            factual=factual,
            factual_vector=factual,
            cf_vectors=cf_unique_opt[0],
            cf_not_optimized_vectors=cf_unique,
            obj_scores=cf_unique_opt[1],
            time_cf=time_cf.total_seconds(),
            time_cf_not_optimized=time_cf_not_optimized.total_seconds(),
            model_pred=model_predict_proba,)


def find_image(
        img: np.ndarray,
        model_predict,
        count_cf: int = 1,
        segmentation: str = 'quickshift',
        params_segmentation: dict = None,
        replace_mode: str = 'blur',
        img_cf_strategy: str = 'nonspecific',
        cf_strategy: str = 'random',
        increase_threshold: (int, float) = None,
        it_max: int = 5000,
        limit_seconds: int = 120,
        ft_change_factor: float = 0.1,
        ft_it_max: int = None,
        size_tabu: (int, float) = 0.5,
        ft_threshold_distance: float = -1.0,
        avoid_back_original: bool = True,
        threshold_changes: int = 100,
        verbose: bool = False) -> _CFImage:
    """
    For an image input and prediction model, finds a counterfactual explanation
    :param img: The original image to be explained
    :type img: np.ndarray
    :param model_predict: Model's function which generates a class probability output
    :type model_predict: object
    :param count_cf: Number of counterfactual explanations to be returned
    :type count_cf: int
    :param segmentation: (optional) Type of segmentation to used in image. Can be: 'quickshift'. Default='quickshift'
    :type segmentation: str
    :param params_segmentation: (optional) Parameters passed to the segmentation algorithm.
    Default={'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
    :type: dict
    :param replace_mode: (optional) Type of replaced image, it can be 'blur', 'mean', 'random' or
    'inpaint'. Default='blur'
    :type replace_mode: str
    :param img_cf_strategy: (optional) Class to be considered in the CF change. Options are 'nonspecific' which changes
     to any CF class other than the factual or 'second_best' which tries to flip the classification
     to the second-highest class for the current factual. Default='nonspecific'
    :type img_cf_strategy: str
    :param cf_strategy: (optional) Strategy to find CF, can be "greedy", "random" or "random-sequential".
    Default='random'
    :type cf_strategy: str
    :param increase_threshold: (optional) Threshold for improvement in CF score in the CF search,
    if the improvement is below that, search will stop. -1 deactivates this check. Default=-1 (greedy) or 1e-05 (random)
    :type increase_threshold: int, float
    :param it_max: (optional) Maximum number of iterations for CF search. Default=5000
    :type it_max: int
    :param limit_seconds: (optional) Time threshold for CF optimization. Default=120
    :type limit_seconds: int
    :param ft_change_factor: (optional) Factor used for numerical features to change their values (e.g. if 0.1 it will
    use, initially, 10% of features' value). Default=0.1
    :type ft_change_factor: float
    :param ft_it_max: (optional) Maximum number of iterations for CF optimization step.
    Default=1000 (greedy), 500 (random)
    :type ft_it_max: int
    :param size_tabu: (optional) (optional) Size of tabu list, if float it's the share of features,
    if int it's the exact number defined. Default=0.5
    :type size_tabu: int, float
    :param ft_threshold_distance: (optional) Threshold for CF optimization enhancement, if improvement is below the
    threshold, the optimization will be stopped. -1 deactivates this check. Default=-1.0
    :type ft_threshold_distance: float
    :param avoid_back_original: (optional) For the greedy strategy, not allows changing back to the original values.
    Default=True
    :type avoid_back_original: bool
    :param threshold_changes: (optional) For the random strategy, threshold for the maximum number of changes to be
    created in the CF finding process. Default=100
    :type threshold_changes: int
    :param verbose: (optional) If True, it will output detailed information of CF finding and optimization steps.
    Default=False
    :type verbose: bool
    :return: Object with CF information
    :rtype: _CFImage
    """

    # Defines the type of data
    cf_data_type = 'image'

    # Some default parameters depend on the select CF finding strategy
    cf_finder = None
    finder_strategy = None
    if cf_strategy == 'random' or cf_strategy == 'random-sequential':
        cf_finder = _random_generator
        if ft_it_max is None:
            ft_it_max = 500
        if avoid_back_original is None:
            avoid_back_original = False
        if cf_strategy == 'random-sequential':
            finder_strategy = 'sequential'
        if increase_threshold is None:
            increase_threshold = 1e-05
            ft_it_max = 500
        if increase_threshold is None:
            increase_threshold = 1e-05
    elif cf_strategy == 'greedy':
        cf_finder = _greedy_generator
        if ft_it_max is None:
            ft_it_max = 1000
        if increase_threshold is None:
            increase_threshold = -1
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

    # First, we need to create the segmentation for image
    if segmentation == 'quickshift':
        segments = gen_quickshift(img, params_segmentation)
    else:
        raise AttributeError(f'segmentation must be "quickshift" and not {segmentation}')

    # Then we create the image to be replaced (considered 0)
    if replace_mode == 'mean':
        replace_img = np.zeros(img.shape)
        replace_img[:, :, 0], replace_img[:, :, 1], replace_img[:, :, 2] = img.mean(axis=(0, 1))
    elif replace_mode == 'blur':
        replace_img = cv2.GaussianBlur(img, (31, 31), 0)
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

    size_tabu = _define_tabu_size(size_tabu, factual)

    # TODO: Make checks

    # Define all features as binary
    feat_types = {fidx: 'cat' for fidx in range(len(factual))}

    # Then, the model must be modified to accept this new kind of data, where 1 means the original values and 0 the
    # modified values
    mic = _adjust_image_model(img, model_predict, segments, replace_img)

    # Get probability values adjusted
    if img_cf_strategy == 'nonspecific':
        mimns = _adjust_multiclass_nonspecific(factual, mic)
    elif img_cf_strategy == 'second_best':
        mimns = _adjust_multiclass_second_best(factual, mic)
    else:
        raise AttributeError(f'img_cf_strategy must be "nonspecific" or "second_best" and not {img_cf_strategy}')

    # Timer now
    time_start = datetime.now()

    # Generate CF using a CF finder
    cf_unique = cf_finder(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
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
        threshold_changes=threshold_changes,
        count_cf=count_cf,
        cf_unique=[],
        verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    if len(cf_unique) == 0:
        logging.log(30, 'No CF found.')
        return _CFImage(
            factual=img,
            factual_vector=factual,
            cf_vectors=[],
            cf_not_optimized_vectors=[],
            obj_scores=[],
            time_cf=time_cf_not_optimized.total_seconds(),
            time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

            _seg_to_img=_seg_to_img,
            segments=segments,
            replace_img=replace_img)

    # Fine tune the counterfactual
    cf_unique_opt = _fine_tuning(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
        factual=factual,
        cf_unique=cf_unique,
        count_cf=count_cf,
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
        limit_seconds=limit_seconds,
        cf_finder=cf_finder,
        avoid_back_original=avoid_back_original,
        threshold_changes=threshold_changes,
        verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    # Create response object
    response_obj = _CFImage(
        factual=img,
        factual_vector=factual,
        cf_vectors=cf_unique_opt[0],
        cf_not_optimized_vectors=cf_unique,
        obj_scores=cf_unique_opt[1],
        time_cf=time_cf.total_seconds(),
        time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

        _seg_to_img=_seg_to_img,
        segments=segments,
        replace_img=replace_img
    )

    return response_obj


def find_text(
        text_input: str,
        textual_classifier,
        count_cf: int = 1,
        word_replace_strategy: str = 'remove',
        cf_strategy: str = 'random',
        increase_threshold: (int, float) = -1,
        it_max: int = 5000,
        limit_seconds: int = 120,
        ft_change_factor: float = 0.1,
        ft_it_max: int = None,
        size_tabu: (int, float) = None,
        ft_threshold_distance: float = None,
        avoid_back_original: bool = False,
        threshold_changes: int = 100,
        verbose: bool = False) -> _CFText:
    """
    For a text input and prediction model, finds a counterfactual explanation
    :param text_input: The original text to be explained
    :type text_input: str
    :param textual_classifier: Model's function which generates a class probability output
    :type textual_classifier: object
    :param count_cf: Number of counterfactual explanations to be returned
    :type count_cf: int
    :param word_replace_strategy: (optional) Strategy to make text modifications. Can be 'remove', which simply
    removes the word or 'antonyms' which replace words by their respective antonyms. Default='remove'
    :type word_replace_strategy: str
    :param cf_strategy: (optional) Strategy to find CF, can be "greedy", "random", or "random-sequential".
    Default: "random"
    :type cf_strategy: str
    :param increase_threshold: (optional) Threshold for improvement in CF score in the CF search,
    if the improvement is below that, search will stop. -1 deactivates this check. Default=-1
    :type increase_threshold: int, float
    :param it_max: (optional) Maximum number of iterations for CF search. Default=5000
    :type it_max: int
    :param limit_seconds: (optional) Time threshold for CF optimization. Default=120
    :type limit_seconds: int
    :param ft_change_factor: (optional) Factor used for numerical features to change their values (e.g. if 0.1 it will
    use, initially, 10% of features' value). Default=0.1
    :type ft_change_factor: float
    :param ft_it_max: (optional) Maximum number of iterations for CF optimization step. Default=2000 (greedy)
    or 100 (random)
    :type ft_it_max: int
    :param size_tabu: (optional) Size of tabu list, if float it's the share of features,
    if int it's the exact number defined. Default= 5 (greedy) or 0.1 (random)
    :type size_tabu: int, float
    :param ft_threshold_distance: (optional) Threshold for CF optimization enhancement, if improvement is below the
    threshold, the optimization will be stopped. -1 deactivates this check. Default=-1 (greedy) 1e-05 (random)
    :type ft_threshold_distance: float
    :param avoid_back_original: (optional) For the greedy strategy, not allows changing back to the original values.
    Default= False
    :type avoid_back_original: bool
    :param threshold_changes: (optional) For the random strategy, threshold for the maximum number of changes to be
    created in the CF finding process. Default=100
    :type threshold_changes: int
    :param verbose: (optional) If True, it will output detailed information of CF finding and optimization steps.
    Default=False
    :type verbose: bool
    :return: Object with CF information
    :rtype: _CFImage
    """

    # Defines the type of data
    cf_data_type = 'text'

    # Select CF finding strategy
    cf_finder = None
    finder_strategy = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
        if ft_it_max is None:
            ft_it_max = 100
        if size_tabu is None:
            size_tabu = 0.1
        if ft_threshold_distance is None:
            ft_threshold_distance = 1e-05
    elif cf_strategy == 'random-sequential':
        cf_finder = _random_generator
        finder_strategy = 'sequential'
        if ft_it_max is None:
            ft_it_max = 100
        if size_tabu is None:
            size_tabu = 0.1
        if ft_threshold_distance is None:
            ft_threshold_distance = 1e-05
    elif cf_strategy == 'greedy':
        cf_finder = _greedy_generator
        if ft_it_max is None:
            ft_it_max = 2000
        if size_tabu is None:
            size_tabu = 5
        if ft_threshold_distance is None:
            ft_threshold_distance = -1.0
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

    # TODO: Make checkers

    # Define type of word replacement strategy
    if word_replace_strategy == 'remove':
        text_words, change_vector, text_replace = _text_to_token_vector(text_input)
    elif word_replace_strategy == 'antonyms':
        text_words, change_vector, text_replace = _text_to_change_vector(text_input)
    else:
        raise AttributeError(f'word_replace_strategy must be "antonyms" or "remove" and not {word_replace_strategy}')

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
    adjusted_encoded_classification = (encoded_text_classification if original_text_classification < 0.5 else
                                       1 - encoded_text_classification)
    if adjusted_encoded_classification != original_text_classification:
        log_warn_text = f'The input factual text has a different classification if compared with the encoded '\
                        f'factual input. This happens because during the encoding process, the text structure '\
                        f'could not be reproduced with 100% fidelity (e.g., extra commas were removed). '
        # If the class is the same, no problem:
        if (adjusted_encoded_classification > 0.5) == (original_text_classification > 0.5):
            log_warn_text += f'This should not be a problem since the classification still the same:\n'
        else:
            # If we had a class change, then the CF is the modification already done
            log_warn_text += f'THE MODIFICATIONS WERE ENOUGH TO GENERATE A COUNTERFACTUAL:\n'

        log_warn_text += f'Original input classification = {original_text_classification}\n' \
                         f'Encoded input classification = {adjusted_encoded_classification}'

        logging.log(30, log_warn_text)

    # Generate OHE parameters if it has OHE variables
    ohe_list, ohe_indexes = _get_ohe_params(factual.iloc[0], True)

    size_tabu = _define_tabu_size(size_tabu, factual.iloc[0])

    # Timer now
    time_start = datetime.now()

    # Generate CF using a CF finder
    cf_unique = cf_finder(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
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
        threshold_changes=threshold_changes,
        count_cf=count_cf,
        cf_unique=[],
        verbose=verbose)

    # Calculate time to generate the not optimized CF
    time_cf_not_optimized = datetime.now() - time_start

    # If no CF was found, return original text, since this may be common, it will not raise errors
    if len(cf_unique) == 0:
        logging.log(30, 'No CF found.')
        return _CFText(
            factual=text_input,
            factual_vector=factual.to_numpy()[0],
            cf_vectors=[],
            cf_not_optimized_vectors=[],
            obj_scores=[],
            time_cf=time_cf_not_optimized.total_seconds(),
            time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

            converter=converter,
            text_replace=text_replace,
        )

    # Fine tune the counterfactual
    cf_unique_opt = _fine_tuning(
        finder_strategy=finder_strategy,
        cf_data_type=cf_data_type,
        factual=factual.iloc[0],
        cf_unique=cf_unique,
        count_cf=count_cf,
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
        limit_seconds=limit_seconds,
        cf_finder=cf_finder,
        avoid_back_original=avoid_back_original,
        threshold_changes=threshold_changes,
        verbose=verbose)

    # Total time to generate the optimized CF
    time_cf = datetime.now() - time_start

    # Create response object
    response_obj = _CFText(
        factual=text_input,
        factual_vector=factual.to_numpy()[0],
        cf_vectors=cf_unique_opt[0],
        cf_not_optimized_vectors=cf_unique,
        obj_scores=cf_unique_opt[1],
        time_cf=time_cf.total_seconds(),
        time_cf_not_optimized=time_cf_not_optimized.total_seconds(),

        converter=converter,
        text_replace=text_replace,
    )

    return response_obj
