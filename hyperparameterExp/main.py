import os
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from typing_extensions import Literal

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
tf.debugging.experimental.disable_dump_debug_info()
# Append previous directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from hyperparameterExp.utils.download_datasets import download_datasets
from hyperparameterExp.utils.download_models import download_models

from hyperparameterExp.utils.experiment_parameters_generator import experiment_parameters
from hyperparameterExp.utils.experiment_model_data_generator import DataModelGenerator

from cfnow.cf_finder import find_tabular, find_text, find_image

global initial_exp_time

# Get current script directory
script_dir = os.path.dirname(os.path.realpath(__file__))

VERBOSE = False

# Get DATA_TYPE, NUM_PARTITIONS and PARTITION_ID, NUM_SAMPLE environment variables
DATA_TYPE = os.environ.get('DATA_TYPE')
NUM_PARTITIONS = int(os.environ.get('NUM_PARTITIONS'))
PARTITION_ID = int(os.environ.get('PARTITION_ID'))
NUM_SAMPLE_PARAMETERS = int(os.environ.get('NUM_SAMPLE_PARAMETERS'))
START_ID = int(os.environ.get('START_ID')) if os.environ.get('START_ID') else 0
NUM_PROCESS = int(os.environ.get('NUM_PROCESS')) if os.environ.get('NUM_PROCESS') else 1

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


def make_experiment(factual, model, cf_strategy, parameters, feat_types=None):


    if VERBOSE:
        print(f'{DATA_TYPE} - {cf_strategy}\nParameters:\n')
        print(pd.Series(parameters))

    feat_types_param = {}
    if DATA_TYPE == 'tabular':
        feat_types_param['feat_types'] = feat_types
    cf_out = cfnow_function(factual, model, cf_strategy=cf_strategy, **parameters, **feat_types_param)

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


dmg = DataModelGenerator(data_type=DATA_TYPE)
# Calculate total number of experiments
# Number of greedy experiments per data point
number_greedy_exp = len(combination_param_greedy_partition)
# Number of random experiments per data point
number_random_exp = len(combination_param_random_partition)
# Number of data points to be tested
number_data_exp = len(dmg.experiment_idx)
total_experiments = (number_greedy_exp + number_random_exp) * number_data_exp

skipped_experiments = 0

# Get the parameters

cf_times = []


class ExperimentIterator:

    len_g_param = len(combination_param_greedy_partition)
    len_r_param = len(combination_param_random_partition)

    def __init__(self):
        # Define the initial state of the iterator
        self.exp_idx = 0
        self.param_idx = 0
        self.params = combination_param_greedy_partition[self.param_idx], 0, 'greedy'
        self.data_model = dmg.next()
        for _ in range(START_ID):
            self.__next__()

    def _next_params(self):
        self.param_idx += 1
        self.exp_idx += 1
        if self.param_idx < self.len_g_param:
            data_idx = self.param_idx
            return combination_param_greedy_partition[data_idx], data_idx,  'greedy'
        elif self.len_g_param <= self.param_idx < self.len_g_param + self.len_r_param:
            data_idx = self.param_idx - self.len_g_param
            return combination_param_greedy_partition[data_idx], data_idx, 'random'
        else:
            # We only change the model when we ran all data points
            self.data_model = dmg.next()
            self.param_idx = 0
            data_idx = self.param_idx
            return combination_param_greedy_partition[data_idx], data_idx, 'greedy'

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_model is None:
            raise StopIteration

        self.return_data = self.exp_idx, self.params, self.data_model

        self.params = self._next_params()

        return self.return_data


def run_exp_process():
    dmg.next()


def print_progress(exp_time):

    cf_times.append(exp_time)
    total_time = round((time.time() - initial_exp_time) / 60, 4)
    remaining_experiments = total_experiments - len(cf_times) - skipped_experiments
    remaining_time = round(sum(cf_times)/sum(cf_times)*remaining_experiments/NUM_PROCESS, 4)

    print(f'\r({DATA_TYPE}) Total time: {total_time} min | '
          f'Estimated Remaining: '
          f'{remaining_time} min',
          flush=True, end='')


def run_experiment_with_parameters(
        experiment_id: int,
        data_exp_id: int,
        data_model: [pd.Series, object, dict],
        cf_strategy: Literal['greedy', 'random'],
        exp_params: dict):
    """
    Run an experiment with the given parameters
    :param experiment_id: The global experiment id
    :param data_exp_id: The specific data id of the experiment
    :param data_model: The data/model from DMG
    :param cf_strategy: CF strategy to use
    :param exp_params: Parameters of the experiment
    :return:
    """

    factual = data_model[0]
    model = data_model[1]
    feat_types = data_model[4]

    if not os.path.exists(f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}'):
        os.makedirs(f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}')

    # Inner loop: for each combination of parameters

    exp_start_time = time.time()
    g_exp_result = make_experiment(factual, model, cf_strategy, exp_params, feat_types)
    exp_time = time.time() - exp_start_time

    g_exp_result['experiment_id'] = data_exp_id

    g_exp_result_pd = pd.DataFrame([g_exp_result])
    # Append pandas dataframe to a pickle file
    g_exp_result_pd.to_pickle(
        f'{script_dir}/Results/{DATA_TYPE}/{PARTITION_ID}/'
        f'{cf_strategy}_{data_exp_id}_{PARTITION_ID}_{experiment_id}.pkl')

    print_progress(exp_time)

experiments = iter(ExperimentIterator())

def exp_thread_run(exp):
    experiment_id = exp[0]
    data_exp_id = exp[1][1]
    data_model = exp[2]
    cf_strategy = exp[1][2]
    exp_params = exp[1][0]

    run_experiment_with_parameters(experiment_id, data_exp_id, data_model, cf_strategy, exp_params)

# Run the experiments in parallel
with ThreadPoolExecutor(max_workers=NUM_PROCESS) as executor:
    initial_exp_time = time.time()
    for exp in experiments:
        executor.submit(exp_thread_run, exp)
        