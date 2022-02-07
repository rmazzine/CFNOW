"""
This module has the function that produces the CF using different types of cf finders and fine-tune.
"""
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import cv2

from ._cf_searchers import _random_generator, _super_sedc
from ._checkers import _check_factual, _check_vars, _check_prob_func
from ._data_standardizer import _get_ohe_params, _seg_to_img
from ._fine_tune import _fine_tuning
from ._model_standardizer import _standardize_predictor, _adjust_model_class, _adjust_image_model, \
    _adjust_image_multiclass_nonspecific, _adjust_image_multiclass_second_best
from ._obj_functions import _obj_manhattan
from ._img_segmentation import gen_quickshift


warnings.filterwarnings("ignore", category=UserWarning)


# TODO: Instead of a step greedy search strategy, create a random initialization to create CF then reduce
# Using Tabu optimization -> This can be done several times until find a good, lowest CF

# TODO: Categorical only datasets have minimum modification equal to 1, we can detect if this happens and stop search


def find_tabular(factual, feat_types, model_predict_proba,
         cf_strategy='greedy', increase_threshold=0, it_max=1000, limit_seconds=30, ft_change_factor=0.1,
         ft_it_max=1000, size_tabu=5, ft_threshold_distance=0.01, has_ohe=False, verbose=False):
    """

    :param factual: The factual point as Pandas DataFrame
    :type factual: pandas.DataFrame
    :param feat_types: A dictionary with {col_name: col_type}, where "col_name" is the name of the column and
    "col_type" can be "num" to indicate numerical continuous features and "cat" to indicate categorical
    :type feat_types: dict
    :param model_predict_proba: Model's function which generates a class probability output
    :type model_predict_proba: object
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
    :param verbose: (optional) If True, it will output detailed information of CF finding and optimization steps.
    Default=False
    :type verbose: bool
    :return: (list) Containing [CF_array, CF_probability, CF_objective_value]
    """

    cf_finder = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
    elif cf_strategy == 'greedy':
        cf_finder = _super_sedc
    if cf_finder is None:
        raise AttributeError(f'cf_strategy must be "random" or "greedy" and not {cf_strategy}')

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
    cf_out = cf_finder(factual=factual,
                       mp1c=mp1c,
                       feat_types=feat_types,
                       it_max=it_max,
                       ft_change_factor=ft_change_factor,
                       ohe_list=ohe_list,
                       ohe_indexes=ohe_indexes,
                       increase_threshold=increase_threshold,
                       tabu_list=None,
                       size_tabu=size_tabu,
                       verbose=verbose)

    if mp1c(np.array([cf_out]))[0] < 0.5:
        raise Warning('No CF found')

    # Fine tune the counterfactual
    cf_out_ft = _fine_tuning(factual=factual,
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
                              verbose=verbose)

    print(_obj_manhattan(np.array(factual), cf_out_ft[0]))
    print(sum(cf_out_ft[0]))

    return cf_out_ft


def find_image(img, model_predict, segmentation='quickshift', params_segmentation=None, cf_strategy='greedy',
               replace_mode='blur', increase_threshold=-1, it_max=1000, limit_seconds=30, ft_change_factor=0.1,
               ft_it_max=1000, size_tabu=None, ft_threshold_distance=0.01, has_ohe=False, verbose=False):
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

    cf_finder = None
    if cf_strategy == 'random':
        cf_finder = _random_generator
    elif cf_strategy == 'greedy':
        cf_finder = _super_sedc
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

    # Green replacement to highlight changes
    green_replace = np.zeros(img.shape)
    green_replace[:,:,1] = 1

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
    mimns = _adjust_image_multiclass_second_best(factual, mic)

    # Generate CF using a CF finder
    cf_out = cf_finder(factual=factual,
                       mp1c=mimns,
                       feat_types=feat_types,
                       it_max=it_max,
                       ft_change_factor=ft_change_factor,
                       ohe_list=[],
                       ohe_indexes=[],
                       increase_threshold=increase_threshold,
                       tabu_list=None,
                       size_tabu=size_tabu,
                       verbose=verbose)

    if mimns(np.array([cf_out]))[0] < 0.5:
        raise Warning('No CF found')

    # Fine tune the counterfactual
    cf_out_ft = _fine_tuning(factual=factual,
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
                              verbose=verbose)

    cf_img = _seg_to_img([cf_out_ft[0]], img, segments, replace_img)[0]
    cf_img_highlight = _seg_to_img([cf_out_ft[0]], img, segments, green_replace)[0]

    return cf_img, cf_img_highlight
