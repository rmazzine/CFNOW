import os

from hyperparameterExp.utils.data import download_and_unzip_data

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Verify if there's a Models folder, if not create it
if not os.path.exists(f'{SCRIPT_DIR}/../Models'):
    os.mkdir(f'{SCRIPT_DIR}/../Models')


def download_models() -> None:
    """
    Download and unzip all models
    :return:
    """
    
    # Download Tabular Models
    # Verify if there's a tabular data folder
    if 'TABULAR_MODELS' not in os.listdir(f'{SCRIPT_DIR}/../Models'):
        download_and_unzip_data(
            save_dir=f'{SCRIPT_DIR}/../',
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='tabular_models.tar.xz',
            folder_name='Models/TABULAR_MODELS')

    # Download Textual Models
    # Verify if there's a textual data folder
    if 'TEXT_MODELS' not in os.listdir(f'{SCRIPT_DIR}/../Models'):
        download_and_unzip_data(
            save_dir=f'{SCRIPT_DIR}/../',
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='imdb_bert.tar.xz',
            folder_name='Models/TEXT_MODELS')

        download_and_unzip_data(
            save_dir=f'{SCRIPT_DIR}/../',
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='twitter_bert.tar.xz',
            folder_name='Models/TEXT_MODELS')
