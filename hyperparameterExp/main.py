import os
import sys
import time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
tf.debugging.experimental.disable_dump_debug_info()
# Append previous directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal

import numpy as np
import pandas as pd

from hyperparameterExp.utils.download_datasets import download_datasets
from hyperparameterExp.utils.download_models import download_models

from hyperparameterExp.utils.experiment_parameters_generator import experiment_parameters
from hyperparameterExp.utils.experiment_model_data_generator import DataModelGenerator

from cfnow.cf_finder import find_tabular, find_text, find_image

# Get current script directory
script_dir = os.path.dirname(os.path.realpath(__file__))

VERBOSE = False

# Get DATA_TYPE, NUM_PARTITIONS and PARTITION_ID, NUM_SAMPLE environment variables
DATA_TYPE = os.environ.get('DATA_TYPE')
NUM_PARTITIONS = int(os.environ.get('NUM_PARTITIONS'))
PARTITION_ID = int(os.environ.get('PARTITION_ID'))
NUM_SAMPLE_PARAMETERS = int(os.environ.get('NUM_SAMPLE_PARAMETERS'))
START_ID = int(os.environ.get('START_ID')) if os.environ.get('START_ID') else 0

# Download data files
download_datasets()

# Download model files
download_models()

# Run experiments
if DATA_TYPE == 'tabular':
    cfnow_function = find_tabular
elif DATA_TYPE == 'image':
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
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


def print_results(factual_prob,
                  cf_prob,
                  cf_not_optimized_prob,
                  cf_time,
                  cf_not_optimized_time,
                  cf_segments,
                  cf_not_optimized_segments,
                  cf_words,
                  cf_not_optimized_words):
    if cf_prob is None:
        print(f'No CF found')
    else:
        print(f'Factual prob: {factual_prob}\n'
              f'CF prob: {cf_prob}\n'
              f'CF-NO prob: {cf_not_optimized_prob}\n'
              f'CF time: {cf_time}\n'
              f'CF-NO time: {cf_not_optimized_time}\n')

        if DATA_TYPE == 'image':
            print(f'CF segments: {cf_segments}\n'
                  f'CF-NO segments: {cf_not_optimized_segments}\n')

        if DATA_TYPE == 'text':
            print(f'CF words: {cf_words}\n'
                  f'CF-NO words: {cf_not_optimized_words}\n')


# Function to calculate L1 distance
def l1_distance(a, b):
    return np.sum(np.abs(a - b))


def make_experiment(factual, model, cf_strategy, parameters):
    def handler(signum, frame):
        raise TimeoutError

    if VERBOSE:
        print(f'{DATA_TYPE} - {cf_strategy}\nParameters:\n')
        print(pd.Series(parameters))

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)

    signal.alarm(300)
    try:
        cf_out = cfnow_function(factual, model, cf_strategy=cf_strategy, **parameters)
    except:
        cf_out = None
    finally:
        signal.alarm(0)

    factual_prob = None
    cf_prob = None
    cf_not_optimized_prob = None
    cf_time = None
    cf_not_optimized_time = None
    cf_segments = None
    cf_not_optimized_segments = None
    cf_words = None
    cf_not_optimized_words = None
    factual_cf_distance = None
    factual_cf_not_optimized_distance = None
    if cf_out is not None:
        if cf_out.cf is not None:
            cf_time = cf_out.time_cf
            cf_not_optimized_time = cf_out.time_cf_not_optimized
            if DATA_TYPE == 'tabular':
                factual_prob = model(np.array([factual]))[0][1]
                cf_prob = model(np.array([cf_out.cf]))[0][1]
                cf_not_optimized_prob = model(np.array([cf_out.cf_not_optimized]))[0][1]
                factual_cf_distance = l1_distance(factual, cf_out.cf)
                factual_cf_not_optimized_distance = l1_distance(factual, cf_out.cf_not_optimized)
            elif DATA_TYPE == 'image':
                factual_prob = np.argmax(model(np.array([factual]))[0])
                cf_prob = np.argmax(model(np.array([cf_out.cf]))[0])
                cf_not_optimized_prob = np.argmax(model(np.array([cf_out.cf_not_optimized]))[0])
                cf_segments = len(cf_out.cf_segments)
                cf_not_optimized_segments = len(cf_out.cf_not_optimized_segments)
                factual_cf_distance = len(cf_out.cf_segments)
                factual_cf_not_optimized_distance = len(cf_out.cf_not_optimized_segments)
            if DATA_TYPE == 'text':
                factual_prob = model(np.array([factual]))[0][0]
                cf_prob = model(np.array([cf_out.cf]))[0][0]
                cf_not_optimized_prob = model(np.array([cf_out.cf_not_optimized]))[0][0]
                cf_words = len(cf_out.cf_replaced_words)
                cf_not_optimized_words = len(cf_out.cf_not_optimized_replaced_words)
                factual_cf_distance = len(cf_out.cf_replaced_words)
                factual_cf_not_optimized_distance = len(cf_out.cf_not_optimized_replaced_words)

    if VERBOSE:
        print('CF Results:\n')
        print_results(factual_prob,
                      cf_prob,
                      cf_not_optimized_prob,
                      cf_time,
                      cf_not_optimized_time,
                      cf_segments,
                      cf_not_optimized_segments,
                      cf_words,
                      cf_not_optimized_words)
        print('############################################################')

    result_out = {
        'data_type': DATA_TYPE,
        'partition_id': PARTITION_ID,
        'cf_strategy': cf_strategy,
        'factual_prob': factual_prob,
        'cf_prob': cf_prob,
        'cf_not_optimized_prob': cf_not_optimized_prob,
        'cf_time': cf_time,
        'cf_not_optimized_time': cf_not_optimized_time,
        'cf_segments': cf_segments,
        'cf_not_optimized_segments': cf_not_optimized_segments,
        'cf_words': cf_words,
        'cf_not_optimized_words': cf_not_optimized_words,
        'factual_cf_distance': factual_cf_distance,
        'factual_cf_not_optimized_distance': factual_cf_not_optimized_distance,
        'parameters': parameters
    }

    return result_out


# Outer loop: for each data and model
dmg = DataModelGenerator(data_type=DATA_TYPE)

# Calculate total number of experiments
# Number of greedy experiments per data point
number_greedy_exp = len(combination_param_greedy_partition)
# Number of random experiments per data point
number_random_exp = len(combination_param_random_partition)
# Number of data points to be tested
number_data_exp = len(dmg.experiment_idx)
total_experiments = (number_greedy_exp + number_random_exp) * number_data_exp
cf_times = []

experiment_id = 0
while True:
    g_data_model = dmg.next()
    if g_data_model is None:
        break

    factual = g_data_model[0]
    model = g_data_model[1]
    feat_types = g_data_model[4]

    if not os.path.exists(f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}'):
        os.makedirs(f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}')

    # Inner loop: for each combination of parameters

    # Greedy Experiments
    partition_g_exp_id = 0
    for g_params in combination_param_greedy_partition:

        if experiment_id < START_ID:
            partition_g_exp_id += 1
            experiment_id += 1
            continue

        g_time_start = time.time()
        g_exp_result = make_experiment(factual, model, 'greedy', g_params)
        g_time_total = time.time() - g_time_start

        g_exp_result['experiment_id'] = partition_g_exp_id

        g_exp_result_pd = pd.DataFrame([g_exp_result])
        # Append pandas dataframe to a pickle file
        g_exp_result_pd.to_pickle(
            f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}/'
            f'greedy_{partition_g_exp_id}_{PARTITION_ID}_{experiment_id}.pkl')

        if VERBOSE:
            print(f'Partition {partition_g_exp_id + 1}/{len(combination_param_greedy_partition)} done')
        partition_g_exp_id += 1
        experiment_id += 1

        cf_times.append(g_time_total)
        print(f'\r({DATA_TYPE}) Total time: {round(sum(cf_times)/60, 4)} min | '
              f'Estimated Remaining: {round(sum(cf_times)/60*total_experiments/len(cf_times), 4)} min',
              flush=True, end='')

    # Random Experiments
    partition_exp_r_id = 0
    for r_params in combination_param_random_partition:

        if experiment_id < START_ID:
            partition_exp_r_id += 1
            experiment_id += 1
            continue

        r_time_start = time.time()
        r_exp_result = make_experiment(factual, model, 'random', r_params)
        r_time_total = time.time() - r_time_start

        r_exp_result['experiment_id'] = partition_exp_r_id

        r_exp_result_pd = pd.DataFrame([r_exp_result])
        # Append pandas dataframe to a pickle file
        r_exp_result_pd.to_pickle(
            f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}/'
            f'random_{partition_exp_r_id}_{PARTITION_ID}_{experiment_id}.pkl')

        if VERBOSE:
            print(f'Partition {partition_exp_r_id + 1}/{len(combination_param_random_partition)} done')
        partition_exp_r_id += 1
        experiment_id += 1

        cf_times.append(r_time_total)
        print(f'\r({DATA_TYPE}) Total time: {round(sum(cf_times)/60, 4)} min | '
              f'Estimated Remaining: {round(sum(cf_times)/60*total_experiments/len(cf_times), 4)} min',
              flush=True, end='')
