import os
import tarfile

import requests


def download_and_unzip_data(save_dir: str, url: str, file_name: str, folder_name: str) -> None:
    """
    Download and unzip a data file
    :param save_dir: Saving directory
    :param url: URL of the data file
    :param file_name: name of the data file
    :param folder_name: output folder name
    :return:
    """

    if not os.path.exists(f'{save_dir}/{folder_name}'):
        os.mkdir(f'{save_dir}/{folder_name}')

    response_download_data = requests.get(f'{url}/{file_name}', stream=True)

    open(f'{save_dir}/{folder_name}/{file_name}', 'wb').write(response_download_data.content)

    tar = tarfile.open(f'{save_dir}/{folder_name}/{file_name}', "r:xz")
    tar.extractall(f'{save_dir}/{folder_name}')
    tar.close()

    os.remove(f'{save_dir}/{folder_name}/{file_name}')