# Using tensorflow-2.8.0 tensorflow-hub-0.12.0 tensorflow-text==2.8.1 tf-models-official
import os
import sys
import hashlib
import random


import numpy as np
import pandas as pd

import tensorflow as tf
# Although it's not used, it's necessary for model load
import tensorflow_text as text

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Append root directory to path
sys.path.append(f'{SCRIPT_DIR}/..')

from textExperiments.utils import timeout
from textExperiments.cf_generators.cfnow import cfnow_greedy, cfnow_random
from textExperiments.cf_generators.limec import make_exp_limec
from textExperiments.cf_generators.shapc import make_exp_shapc


MEMORY_LIMIT = 4 * 1024

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])

random.seed(42)

skip_found = True

cf_generators_experiment = {
    'cfnow_greedy': cfnow_greedy,
    'cfnow_random': cfnow_random,
    'limec': make_exp_limec,
    'shapc': make_exp_shapc,
}


# Count the number the files in all directories inside a directory
def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


TOTAL_EXPERIMENTS = count_files(f'{SCRIPT_DIR}/Datasets')*len(cf_generators_experiment)

if len(sys.argv) == 3:
    START_ID = int(sys.argv[1])
    END_ID = int(sys.argv[2])
else:
    START_ID = 0
    END_ID = int(TOTAL_EXPERIMENTS)


if __name__ == '__main__':
    exp_id = 0
    current_model = None
    # Select IMDB (experiment_dataset = 'imdb') or Twitter (experiment_dataset = 'twitter') experiments
    for experiment_dataset, class_data in [('imdb', 'pos'), ('imdb', 'neg'), ('twitter', 'pos'), ('twitter', 'neg')]:

        # Load model
        if experiment_dataset != current_model:
            model = tf.keras.models.load_model(f'{SCRIPT_DIR}/Models/{experiment_dataset}_bert')
            current_model = experiment_dataset

            def textual_classifier(input_texts):
                return tf.sigmoid(model(tf.constant(input_texts))).numpy()

        # Load data
        files_path = os.listdir(f'{SCRIPT_DIR}/Datasets/exp_files_{experiment_dataset}_{class_data}/')
        files_path = [
            f'{SCRIPT_DIR}/Datasets/exp_files_{experiment_dataset}_{class_data}/{file_path}' for file_path in files_path]
        files_path.sort()

        for file_data in files_path:
            text_input = open(file_data, 'r').read()

            # The experiment hash is based on the original text
            experiment_hash = hashlib.sha256(text_input.encode('utf-8')).hexdigest()

            factual_class = textual_classifier(np.array([text_input]))

            for algorithm_name, cf_generator in cf_generators_experiment.items():

                if START_ID > exp_id:
                    print(f'Skipping Experiment {exp_id}')
                    exp_id += 1
                    continue
                if exp_id >= END_ID:
                    sys.exit(0)
                exp_id += 1

                # Verify if the experiment has already been done
                if skip_found:
                    if algorithm_name not in ['cfnow_greedy', 'cfnow_random']:
                        if os.path.exists(f'{SCRIPT_DIR}/Results/{experiment_hash}_{algorithm_name}.pkl'):
                            print(f'Skipping Experiment {exp_id - 1} ', algorithm_name)
                            continue
                    else:
                        if os.path.exists(
                                f'{SCRIPT_DIR}/Results/{experiment_hash}_{algorithm_name}_optimized.pkl') and \
                                os.path.exists(
                                    f'{SCRIPT_DIR}/Results/{experiment_hash}_{algorithm_name}_not_optimized.pkl'):
                            print(f'Skipping Experiment {exp_id - 1} ', algorithm_name)
                            continue
                try:
                    @timeout(600)
                    def make_cf ():
                        return cf_generator(text_input, textual_classifier, factual_class, model)

                    cf_results = make_cf()

                except Exception as e:
                    print(f'Error in experiment {exp_id - 1} ', algorithm_name, e)
                    if algorithm_name in ['cfnow_greedy', 'cfnow_random']:
                        cf_results = [[None, None, None, None, None], [None, None, None, None, None]]
                    else:
                        cf_results = [[None, None, None, None, None]]

                for r_idx, result in enumerate(cf_results):

                    cf_text, cf_html_highlight, cf_class, list_rem_words, cf_time = result

                    complement_algorithm_name = ''
                    if algorithm_name in ['cfnow_greedy', 'cfnow_random']:
                        if r_idx == 0:
                            complement_algorithm_name = '_optimized'
                        else:
                            complement_algorithm_name = f'_not_optimized'

                    output_data = {
                        'experiment_hash': experiment_hash,
                        'algorithm': algorithm_name + complement_algorithm_name,
                        'factual_text': text_input,
                        'factual_class': factual_class[0][0],
                        'cf_text': cf_text,
                        'cf_html_highlight': cf_html_highlight,
                        'cf_class': cf_class,
                        'list_rem_words': list_rem_words,
                        'cf_time': cf_time,
                    }

                    print(f'Experiment {exp_id - 1} '
                          f'{output_data["algorithm"]} - Factual score {factual_class[0][0]} - CF score {cf_class}')

                    pd.DataFrame([output_data]).to_pickle(
                        f'{SCRIPT_DIR}/Results/{experiment_hash}_{algorithm_name + complement_algorithm_name}.pkl')
