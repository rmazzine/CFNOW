import os

from hyperparameterExp.utils.data import download_and_unzip_data

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Verify if there's a Datasets folder, if not create it
if not os.path.exists(f'{SCRIPT_DIR}/Datasets'):
    os.mkdir(f'{SCRIPT_DIR}/Datasets')


def download_datasets() -> None:
    """
    Download and unzip all datasets
    :return:
    """

    # Download Tabular Data
    # Verify if there's a tabular data folder
    if 'TABULAR_DATA' not in os.listdir(f'{SCRIPT_DIR}/Datasets'):
        download_and_unzip_data(
            script_dir=SCRIPT_DIR,
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='exp_files_tabular.tar.xz',
            folder_name='Datasets/TABULAR_DATA')

    # Download Image Data
    # Verify if there's an image data folder
    if 'IMAGE_DATA' not in os.listdir(f'{SCRIPT_DIR}/Datasets'):
        download_and_unzip_data(
            script_dir=SCRIPT_DIR,
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='CFNOW_image_data.tar.xz',
            folder_name='Datasets/IMAGE_DATA')

    # Download Textual Data
    # Verify if there's a textual data folder
    if 'TEXT_DATA' not in os.listdir(f'{SCRIPT_DIR}/Datasets'):
        download_and_unzip_data(
            script_dir=SCRIPT_DIR,
            url='https://objectstorage.us-ashburn-1.oraclecloud.com/n/idaknh7ztshz/b/CFNOW_Data/o',
            file_name='exp_files_nlp.tar.xz',
            folder_name='Datasets/TEXT_DATA')

