import copy
import random
import itertools

TABULAR_EXPERIMENTS = {
    'increase_threshold': [-1.0, 0.00001],
    'it_max': [100, 1000, 5000],
    'limit_seconds': [10, 30, 120],
    'ft_change_factor': [0.1, 0.5],
    'ft_it_max': [100, 1000, 5000],
    'size_tabu': [5, 0.1, 0.2, 0.5, 0.9],
    'ft_threshold_distance': [-1.0, 0.00001],
    'avoid_back_original': [True, False],
    'threshold_changes': [100, 1000],
}

IMAGE_EXPERIMENTS = {
    'increase_threshold': [-1.0, 0.00001],
    'it_max': [100, 500, 1000],
    'limit_seconds': [10, 30, 120],
    'ft_it_max': [100, 500, 1000],
    'size_tabu': [5, 0.1, 0.2, 0.5, 0.9],
    'ft_threshold_distance': [-1.0, 0.00001],
    'avoid_back_original': [True, False],
    'threshold_changes': [100, 1000],
}

TEXT_EXPERIMENTS = {
    'increase_threshold': [-1.0, 0.00001],
    'it_max': [100, 1000, 2000],
    'limit_seconds': [10, 30, 120],
    'ft_it_max': [100, 1000, 2000],
    'size_tabu': [5, 0.1, 0.2, 0.5, 0.9],
    'ft_threshold_distance': [-1.0, 0.00001],
    'avoid_back_original': [True, False],
    'threshold_changes': [100, 1000],
}


# Function which takes a dictionary with a list of values and returns all possible combinations of the values
def get_combinations(experiment_parameters):
    return list(itertools.product(*experiment_parameters.values()))


# Greedy experiments do not include the 'threshold_changes' parameter
# and Random experiments do not include the 'avoid_back_original' parameter
def experiment_parameters(data_type, cf_strategy, n_sample):

    # Set seed for random number generator
    random.seed(42)

    experiment_parameters_copy = copy.deepcopy(experiment_parameters)

    if data_type == 'tabular':
        experiment_parameters_copy = copy.deepcopy(TABULAR_EXPERIMENTS)
    if data_type == 'image':
        experiment_parameters_copy = copy.deepcopy(IMAGE_EXPERIMENTS)
    if data_type == 'text':
        experiment_parameters_copy = copy.deepcopy(TEXT_EXPERIMENTS)

    if cf_strategy == 'greedy':
        del experiment_parameters_copy['threshold_changes']
    if cf_strategy == 'random':
        del experiment_parameters_copy['avoid_back_original']

    out_experiment_parameters = get_combinations(experiment_parameters_copy)
    total_combination = len(out_experiment_parameters)
    # Get a sample, without replacement, of the possible combinations
    out_experiment_parameters = random.sample(out_experiment_parameters, n_sample)

    out_experiment_parameters = [{key: value for key, value in zip(experiment_parameters_copy.keys(), value)}
                                 for value in out_experiment_parameters]

    return total_combination, out_experiment_parameters
