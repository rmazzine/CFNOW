import os
import signal
import tarfile
from functools import wraps

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


class TimeoutError(Exception): pass


# IMPORTANT: this is not thread-safe
def timeout(seconds, error_message='Function call timed out'):
    def _handle_timeout(signum, frame):
        raise TimeoutError(error_message)

    def decorated(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorated
