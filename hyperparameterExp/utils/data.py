import os
import tarfile

import requests


def download_and_unzip_data(script_dir: str, url: str, file_name: str, folder_name: str) -> None:
    """
    Download and unzip a data file
    :param script_dir: Current script directory
    :param url: URL of the data file
    :param file_name: name of the data file
    :param folder_name: output folder name
    :return:
    """

    if not os.path.exists(f'{script_dir}/{folder_name}'):
        os.mkdir(f'{script_dir}/{folder_name}')

    response_download_data = requests.get(f'{url}/{file_name}', stream=True)

    open(f'{script_dir}/{folder_name}/{file_name}', 'wb').write(response_download_data.content)

    tar = tarfile.open(f'{script_dir}/{folder_name}/{file_name}', "r:xz")
    tar.extractall(f'{script_dir}/{folder_name}')
    tar.close()

    os.remove(f'{script_dir}/{folder_name}/{file_name}')
