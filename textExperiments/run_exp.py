import os
import sys
from subprocess import Popen

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Append root directory to path
sys.path.append(f'{SCRIPT_DIR}/..')

# To download necessary extra packages
from cfnow.cf_finder import find_text

from textExperiments.utils import download_and_unzip_data
MODEL_SUFFIX = 'bert'

BUCKET_URL = 'https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o'

NUM_PROCESS = 1
if len(sys.argv) == 2:
    NUM_PROCESS = int(sys.argv[1])

if not os.path.exists(f'{SCRIPT_DIR}/Datasets'):
    os.mkdir(f'{SCRIPT_DIR}/Datasets')

    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'exp_files_nlp.tar.xz',
        f'Datasets/')

if not os.path.exists(f'{SCRIPT_DIR}/Models'):
    os.mkdir(f'{SCRIPT_DIR}/Models')

if MODEL_SUFFIX == 'bert' and not os.path.exists(f'{SCRIPT_DIR}/Models/imdb_bert'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'imdb_bert.tar.xz',
        f'Models/')
elif MODEL_SUFFIX == 'electra_small' and not os.path.exists(f'{SCRIPT_DIR}/Models/imdb_electra_small'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'imdb_electra_small.tar.gz',
        f'Models/')
elif MODEL_SUFFIX == 'experts_wiki_books' and not os.path.exists(f'{SCRIPT_DIR}/Models/imdb_experts_wiki_books'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'imdb_experts_wiki_books.tar.gz',
        f'Models/')
elif MODEL_SUFFIX == 'talking-heads_base' and not os.path.exists(f'{SCRIPT_DIR}/Models/imdb_talking-heads_base'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'imdb_talking-heads_base.tar.gz',
        f'Models/')

if MODEL_SUFFIX == 'bert' and not os.path.exists(f'{SCRIPT_DIR}/Models/twitter_bert'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'twitter_bert.tar.xz',
        f'Models/')
elif MODEL_SUFFIX == 'electra_small' and not os.path.exists(f'{SCRIPT_DIR}/Models/twitter_electra_small'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'twitter_electra_small.tar.gz',
        f'Models/')
elif MODEL_SUFFIX == 'experts_wiki_books' and not os.path.exists(f'{SCRIPT_DIR}/Models/twitter_experts_wiki_books'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'twitter_experts_wiki_books.tar.gz',
        f'Models/')
elif MODEL_SUFFIX == 'talking-heads_base' and not os.path.exists(f'{SCRIPT_DIR}/Models/twitter_talking-heads_base'):
    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        BUCKET_URL,
        'twitter_talking-heads_base.tar.gz',
        f'Models/')

if not os.path.exists(f'{SCRIPT_DIR}/Results'):
    os.mkdir(f'{SCRIPT_DIR}/Results')

from textExperiments.exp import TOTAL_EXPERIMENTS

# Get a linearly spaced list of numbers from 0 to TOTAL_EXPERIMENTS with NUM_PROCESS elements
# and convert it to a list of tuples
list_of_tuples = list(zip(list(map(int, np.linspace(0, TOTAL_EXPERIMENTS, NUM_PROCESS + 1)[:-1])),
                          list(map(int, np.linspace(0, TOTAL_EXPERIMENTS, NUM_PROCESS + 1)[1:]))))

for start, end in list_of_tuples:
    Popen(['python3', f'{SCRIPT_DIR}/exp.py',
          str(start), str(end), MODEL_SUFFIX])
