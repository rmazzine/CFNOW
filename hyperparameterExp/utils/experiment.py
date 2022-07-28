import numpy as np

from cfnow.cf_finder import find_tabular, find_text, find_image


def make_experiment(
        factual, model, data_type, cf_strategy,
        increase_threshold, it_max, limit_seconds, ft_change_factor, ft_it_max, size_tabu, ft_threshold_distance,
        avoid_back_original, threshold_changes):
    """
    Make an experiment using the CFNOW library, and returns the following relevant information:
    * If a CF was found
    * The number of modifications made to the CF
    * Distance between the original CF and the CF found
    * The time to generate the CF
    :return:
    """

    if data_type == 'tabular':
        cf_result = find_tabular(
            factual=factual,
            model_predict_proba=model,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes)

    elif data_type == 'image':
        cf_result = find_image(
            img=factual,
            model_predict=model,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes)

    elif data_type == 'text':
        cf_result = find_text(
            text_input=factual,
            textual_classifier=model,
            cf_strategy=cf_strategy,
            increase_threshold=increase_threshold,
            it_max=it_max,
            limit_seconds=limit_seconds,
            ft_change_factor=ft_change_factor,
            ft_it_max=ft_it_max,
            size_tabu=size_tabu,
            ft_threshold_distance=ft_threshold_distance,
            avoid_back_original=avoid_back_original,
            threshold_changes=threshold_changes)

    else:
        raise ValueError(f'Data type {data_type} not supported')

    # Process the result
    factual_vector_np = np.array(cf_result.factual_vector)

    if cf_result.cf_vector is not None:
        cf_vector_np = np.array(cf_result.cf_vector)

        found_cf = True
        num_modifications_cf = (np.array(factual_vector_np) != np.array(cf_vector_np)).sum()
        distance_cf = np.linalg.norm(factual_vector_np - cf_vector_np)
        time_cf = cf_result.time_cf
    else:
        found_cf = False
        num_modifications_cf = None
        distance_cf = None
        time_cf = None

    if cf_result.cf_not_optimized_vector is not None:
        cf_not_optimized_vector_np = np.array(cf_result.cf_not_optimized_vector)

        found_cf_not_optimized = True
        num_modifications_cf_not_optimized = (np.array(factual_vector_np) != np.array(cf_not_optimized_vector_np)).sum()
        distance_cf_not_optimized = np.linalg.norm(factual_vector_np - cf_not_optimized_vector_np)
        time_cf_not_optimized = cf_result.time_cf_not_optimized
    else:
        found_cf_not_optimized = False
        num_modifications_cf_not_optimized = None
        distance_cf_not_optimized = None
        time_cf_not_optimized = None

    return {
        'found_cf': found_cf,
        'num_modifications_cf': num_modifications_cf,
        'distance_cf': distance_cf,
        'time_cf': time_cf,
        'found_cf_not_optimized': found_cf_not_optimized,
        'num_modifications_cf_not_optimized': num_modifications_cf_not_optimized,
        'distance_cf_not_optimized': distance_cf_not_optimized,
        'time_cf_not_optimized': time_cf_not_optimized,
    }
