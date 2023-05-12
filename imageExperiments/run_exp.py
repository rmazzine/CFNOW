import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Append root directory to path
sys.path.append(f'{SCRIPT_DIR}/..')

from imageExperiments.utils import download_and_unzip_data

if not os.path.exists(f'{SCRIPT_DIR}/Datasets'):
    os.mkdir(f'{SCRIPT_DIR}/Datasets')

    download_and_unzip_data(
        f'{SCRIPT_DIR}/',
        'https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
        'CFNOW_image_exp_data.tar.xz',
        f'')


from imageExperiments.exp import run_experiment

run_experiment()
