"""
This script creates an iterator that generates data and model for the experiment.
"""

import os

import numpy as np
import pandas as pd
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub

# Although it's not used, it's necessary for model load
import tensorflow_text as text

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f'{SCRIPT_DIR}/../Datasets'
MODEL_DIR = f'{SCRIPT_DIR}/../Models'

# Define image shape
IMAGE_SHAPE = (224, 224)


def tabular_data_model_paths():
    """
    Generates paths to the tabular data and model
    For data: takes the first n first rows of the datasets indicated below:
    0_Adult: 1 || 1_Adult: 1
    0_BalanceScale: 1 || 1_BalanceScale: 1
    0_BCW: 1 || 1_BCW: 1
    0_CarEvaluation: 1 || 1_CarEvaluation: 1
    0_Chess: 1 || 1_Chess: 1
    0_CMSC: 1 || 1_CMSC: 1
    0_DefaultOfCCC: 1 || 1_DefaultOfCCC: 1
    0_Ecoli: 1 || 1_Ecoli: 1
    1_HayesRoth: 1 || 2_HayesRoth: 1
    0_InternetAdv: 1 || 1_InternetAdv: 1
    0_Iris: 1 || 1_Iris: 1
    1_ISOLET: 1 || 2_ISOLET: 1
    1_Lymphography: 1 || 2_Lymphography: 1
    0_MagicGT: 1 || 1_MagicGT: 1
    0_Nursery: 1 || 1_Nursery: 1
    1_PBC: 1 || 2_PBC: 1
    1_SDD: 1 || 2_SDD: 1
    0_SoybeanSmall: 1 || 1_SoybeanSmall: 1
    0_StatlogGC: 1 || 1_StatlogGC: 1
    0_StudentPerf: 1 || 1_StudentPerf: 1
    0_TicTacToe: 1 || 1_TicTacToe: 1
    1_Wine: 1 || 2_Wine: 1
    For model: Takes the model for each respective dataset
    :return:
    """
    data_n_rows = {'0_Adult': 2, '1_Adult': 2, '0_BalanceScale': 2, '1_BalanceScale': 2, '0_BCW': 2, '1_BCW': 2,
                   '0_CarEvaluation': 1, '1_CarEvaluation': 1, '0_Chess': 1, '1_Chess': 1, '0_CMSC': 1, '1_CMSC': 1,
                   '0_DefaultOfCCC': 1, '1_DefaultOfCCC': 1, '0_Ecoli': 1, '1_Ecoli': 1, '1_HayesRoth': 1,
                   '2_HayesRoth': 1, '0_InternetAdv': 1, '1_InternetAdv': 1, '0_Iris': 1, '1_Iris': 1, '1_ISOLET': 1,
                   '2_ISOLET': 1, '1_Lymphography': 1, '2_Lymphography': 1, '0_MagicGT': 1, '1_MagicGT': 1,
                   '0_Nursery': 1, '1_Nursery': 1, '1_PBC': 1, '2_PBC': 1, '1_SDD': 1, '2_SDD': 1,
                   '0_SoybeanSmall': 1, '1_SoybeanSmall': 1, '0_StatlogGC': 1, '1_StatlogGC': 1,
                   '0_StudentPerf': 1, '1_StudentPerf': 1, '0_TicTacToe': 1, '1_TicTacToe': 1, '1_Wine': 1,
                   '2_Wine': 1}

    array_model_data_paths = []
    for dataset, n in data_n_rows.items():
        for r_idx in range(n):
            array_model_data_paths.append([
                f'{DATA_DIR}/TABULAR_DATA/{dataset}.csv',
                f'{MODEL_DIR}/TABULAR_MODELS/{dataset.split("_")[1]}.h5',
                r_idx])

    return array_model_data_paths


def image_data_paths():
    """
    Generates paths to the image file
    For data: takes the first n image files of the classes indicated below:
    acoustic guitars: 4
    barrows: 2
    beach wagons: 2
    beacons: 2
    box turtles: 2
    Chihuahuas: 2
    churchs: 2
    container ships: 2
    electric guitars: 2
    envelopes: 2
    espresso makers: 2
    fire engines: 2
    military uniforms: 2
    missiles: 2
    mouses: 2
    pencil sharpeners: 2
    Polaroid cameras: 2
    revolvers: 2
    rifles: 2
    rugby balls: 2
    soccer balls: 2
    tiger cats: 2
    tigers: 2
    toasters: 2
    :return:
    """
    data_n_rows = {'acoustic guitars': 4, 'barrows': 2, 'beach wagons': 2, 'beacons': 2, 'box turtles': 2,
                   'Chihuahuas': 2, 'churchs': 2, 'container ships': 2, 'electric guitars': 2, 'envelopes': 2,
                   'espresso makers': 2, 'fire engines': 2, 'military uniforms': 2, 'missiles': 2, 'mouses': 2,
                   'pencil sharpeners': 2, 'Polaroid cameras': 2, 'revolvers': 2, 'rifles': 2, 'rugby balls': 2,
                   'soccer balls': 2, 'tiger cats': 2, 'tigers': 2, 'toasters': 2}

    array_data_paths = []
    for dataset, n in data_n_rows.items():
        # Get file paths of a directory
        file_paths = [
            f'{DATA_DIR}/IMAGE_DATA/{dataset}/{file}' for file in
            os.listdir(f'{DATA_DIR}/IMAGE_DATA/{dataset}')]
        file_paths.sort()

        for r_idx in range(n):
            array_data_paths.append([
                file_paths[r_idx],
                'mobile_net_v2_small',
                r_idx])

    return array_data_paths


def text_data_model_paths():
    """
    Generates paths to the text data and model
    For data: takes the first n text of the files indicated below:
    imdb_neg: 15 | imd_pos: 10
    twitter_neg 10 | twitter_neg: 15
    :return:
    """

    data_n_rows = {'imdb_neg': 15, 'imdb_pos': 10, 'twitter_neg': 10, 'twitter_pos': 15}
    array_model_data_paths = []
    for dataset, n in data_n_rows.items():

        # Get file paths of a directory
        file_paths = [
            f'{DATA_DIR}/TEXT_DATA/exp_files_{dataset}/{file}' for file in
            os.listdir(f'{DATA_DIR}/TEXT_DATA/exp_files_{dataset}')]
        file_paths.sort()

        for r_idx in range(n):
            array_model_data_paths.append([
                file_paths[r_idx],
                f'{MODEL_DIR}/TEXT_MODELS/{dataset.split("_")[0]}_bert',
                r_idx])

    return array_model_data_paths


# Load a TF model
def load_tf_model(model_path: str, memory_limit: int) -> tf.keras.Model:
    """
    Load a TF model
    :param model_path: path to the model
    :param memory_limit: memory limit in MB
    :return: the model
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

    model = tf.keras.models.load_model(model_path)

    return model


# Load MobileNetV2 model
def load_mobilenetv2_model(memory_limit: int) -> tf.keras.Model:
    """
    :param memory_limit: memory limit in MB
    Load MobileNetV2 model
    :return: the model
    """
    # Import model
    classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

    # Configure classifier
    model = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,), trainable=True)
    ])

    return model


class DataModelGenerator:
    """
    This class generates a data point and model when called DataModelGenerator().next()
    each time .next() is called, the class verifies if the model has changed, if yes, it loads to the
    internal variable self.model the new model (consequently excluding the old model).
    The experiment_idx is a list of lists, each list contains the path to the data and the path to the model and index
    of the row to be used. [path_data, path_model, idx_row]
    """

    def __init__(self, data_type: str, gpu_memory_fraction: int = 4000,  idx_list: list = None):
        """
        Initializes the class
        :param data_type: The data type to be used, can be 'tabular' or 'image' or 'text'
        :param idx_list: The list of indexes to be used, if None, all the indexes are used
        """
        self.current_idx = 0

        self.data_type = data_type
        self.gpu_memory_fraction = gpu_memory_fraction
        if data_type == 'tabular':
            self.experiment_idx = tabular_data_model_paths()
        elif data_type == 'image':
            self.experiment_idx = image_data_paths()
        elif data_type == 'text':
            self.experiment_idx = text_data_model_paths()
        else:
            raise ValueError(f'Data type {data_type} not supported')

        self.idx_list = idx_list if idx_list is not None else [*range(len(self.experiment_idx))]

        self.model = None
        self.current_model_path = None

    def load_model(self, model_path):
        """
        Loads the model to the internal variable self.model
        :return:
        """
        if model_path != self.current_model_path:
            if self.data_type == 'tabular':
                self.model = load_tf_model(model_path, self.gpu_memory_fraction)
            elif self.data_type == 'image':
                self.model = load_mobilenetv2_model(self.gpu_memory_fraction)
            elif self.data_type == 'text':
                self.model = load_tf_model(model_path, self.gpu_memory_fraction)
            else:
                raise ValueError(f'Data type {self.data_type} not supported')
            self.current_model_path = model_path

    def next(self):
        """
        Returns the next data point and model
        :return: data point and model
        """
        if self.current_idx == len(self.idx_list):
            raise StopIteration
        else:
            data_path, model_path, idx_row = self.experiment_idx[self.idx_list[self.current_idx]]
            self.current_idx += 1
            if self.data_type == 'tabular':
                row = pd.read_csv(data_path, nrows=1, skiprows=idx_row)
            elif self.data_type == 'image':
                # Make adjustments for image work in the classifier
                img = Image.open(f'{data_path}')
                img = img.resize(IMAGE_SHAPE)
                img = np.array(img) / 255.0
                row = img
            elif self.data_type == 'text':
                row = open(data_path, 'r').readlines()
            else:
                raise AttributeError('Data type not supported')

            self.load_model(model_path)

            return row, self.model

a = 1