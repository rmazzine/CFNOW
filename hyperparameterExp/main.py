import os

import numpy as np

from hyperparameterExp.utils.download_datasets import download_datasets
from hyperparameterExp.utils.download_models import download_models

from hyperparameterExp.utils.experiment_parameters_generator import experiment_parameters
from hyperparameterExp.utils.experiment_model_data_generator import DataModelGenerator

from cfnow.cf_finder import find_tabular, find_text, find_image

# Get DATA_TYPE, NUM_PARTITIONS and PARTITION_ID, NUM_SAMPLE environment variables
DATA_TYPE = os.environ.get('DATA_TYPE')
NUM_PARTITIONS = int(os.environ.get('NUM_PARTITIONS'))
PARTITION_ID = int(os.environ.get('PARTITION_ID'))
NUM_SAMPLE_PARAMETERS = int(os.environ.get('NUM_SAMPLE_PARAMETERS'))

# Download data files
download_datasets()

# Download model files
download_models()

# Run experiments

if DATA_TYPE == 'tabular':
    cfnow_function = find_tabular
elif DATA_TYPE == 'image':
    cfnow_function = find_image
elif DATA_TYPE == 'text':
    cfnow_function = find_text


# Greedy search parameters
total_combination_param_greedy, combination_param_greedy = \
    experiment_parameters(data_type=DATA_TYPE, cf_strategy='greedy', n_sample=NUM_SAMPLE_PARAMETERS)
# Random search parameters
total_combination_param_random, combination_param_random = \
    experiment_parameters(data_type=DATA_TYPE, cf_strategy='random', n_sample=NUM_SAMPLE_PARAMETERS)

percentage_possible_combinations = round(
    (NUM_SAMPLE_PARAMETERS*2.0/(total_combination_param_greedy + total_combination_param_random)) * 100, 2)

print(f'The sample represents '
      f'{percentage_possible_combinations}'
      f'% of the total combinations')

partition_greedy_idx = np.array_split(range(len(combination_param_greedy)), NUM_PARTITIONS)[PARTITION_ID - 1]
partition_random_idx = np.array_split(range(len(combination_param_random)), NUM_PARTITIONS)[PARTITION_ID - 1]

combination_param_greedy_partition = [combination_param_greedy[i] for i in partition_greedy_idx]
combination_param_random_partition = [combination_param_random[i] for i in partition_random_idx]

# Greedy Experiments

# Outer loop: for each data and model
dmg = DataModelGenerator(data_type=DATA_TYPE)
while True:
    g_data_model = dmg.next()
    if g_data_model is None:
        break

    # Inner loop: for each combination of parameters
    for g_params in combination_param_greedy_partition:
        print(f'Running hyperparameter search with parameters: {g_params}')
        print(g_data_model[0][:50])
        print(g_data_model[2])
        print(g_data_model[3])
        print(g_data_model[4])
        print(g_data_model[5])


        factual = g_data_model[0]
        model = g_data_model[1]
        feat_types = g_data_model[4]
        cf_out = cfnow_function(factual, model, cf_strategy='greedy', **g_params)

        # print(len(cf_out.cf_replaced_words))
        # print(len(cf_out.cf_not_optimized_replaced_words))
        print('\n')



a = 1


# tabular_data_model_data_generator = DataModelGenerator(data_type='tabular')
# for params in tabular_data_exp_params:
#     factual, model, data_path, model_path, feat_types, idx_row = tabular_data_model_data_generator.next()

# Create report
