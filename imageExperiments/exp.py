import os
import sys
import hashlib
from contextlib import contextmanager

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import tensorflow_hub as hub
from skimage.segmentation import quickshift

from imageExperiments.cf_generators.cfnow import CFNOWGreedyGenerator, CFNOWRandomGenerator
from imageExperiments.cf_generators.blur import BlurGenerator
from imageExperiments.cf_generators.occlusion import OcclusionGenerator
from imageExperiments.cf_generators.sedc import SEDCGenerator
from imageExperiments.cf_generators.limec import LIMECGenerator
from imageExperiments.cf_generators.shapc import SHAPCGenerator


import warnings


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cf_generators = {
    'cfnow_greedy': CFNOWGreedyGenerator,
    'cfnow_random': CFNOWRandomGenerator,
    'blur': BlurGenerator,
    'occlusion': OcclusionGenerator,
    'sedc': SEDCGenerator,
    'limec': LIMECGenerator,
    'shapc': SHAPCGenerator
}


# Import model
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

# Define image shape
IMAGE_SHAPE = (224, 224)

# Get labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Configure classifier
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,), trainable=True)
])

img_files_path = []
image_folders = os.listdir(f'{SCRIPT_DIR}/Datasets')
for img_f in image_folders:
    image_names = os.listdir(f'{SCRIPT_DIR}/Datasets/{img_f}')

    for img_n in image_names:
        img_files_path.append(f'{SCRIPT_DIR}/Datasets/{img_f}/{img_n}')


def run_experiment():
    img_hashes = []
    for idx, img_path in enumerate(img_files_path):

        # Print Current idx and image path
        print(idx, img_path)

        image_class = img_path.split('/')[-2]

        # Make adjustments for image work in the classifier
        img_filename = img_path.split('/')[-1]
        img = Image.open(f'{img_path}')
        img = img.resize(IMAGE_SHAPE)
        img = np.array(img) / 255.0

        img_hash = hashlib.sha256(img).hexdigest()
        # Assert that the image is not repeated
        assert img_hash not in img_hashes
        img_hashes.append(img_hash)

        # Get blurred replace image
        replace_img = cv2.GaussianBlur(img, (31, 31), 0)

        # Get segments
        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
        img_segments = quickshift(img, **params_seg)

        # Get original prediction
        original_prediction = classifier.predict(np.array([img]))
        original_prediction_idx = np.argmax(original_prediction)
        original_classification = imagenet_labels[original_prediction_idx]

        for generator_name, generator in cf_generators.items():

            try:
                list_solutions = []
                list_solution_names = []
                generator_instance = generator(
                    img, classifier, original_prediction_idx, replace_img, imagenet_labels, generator_name, img_hash)

                # Skip the experiment if the flag is set
                if generator_instance.skip_experiment:
                    print(f'Skipping experiment for {generator_name} and {img_filename}')
                    continue

                with suppress_stdout():
                    if generator_name not in ['cfnow_greedy', 'cfnow_random']:
                        list_solutions.append(generator_instance.explain())
                        list_solution_names.append(generator_name)

                    else:
                        for cfnow_type, solution in zip(['optimized', 'not_optimized'], generator_instance.explain()):
                            list_solutions.append(solution)
                            list_solution_names.append(generator_name + '_' + cfnow_type)

                for solution, solution_name in zip(list_solutions, list_solution_names):
                    cf_classificaiton, exp_time, segments_in_explanation = solution
                    # Save solution
                    pd.DataFrame([{
                        'image_hash': img_hash,
                        'original_classification': original_classification,
                        'cf_classification': cf_classificaiton,
                        'time': exp_time,
                        'segments_in_explanation': segments_in_explanation,
                        'generator': solution_name
                    }]).to_pickle(f'{SCRIPT_DIR}/Results/{img_hash}_{solution_name}.pkl')
                    if cf_classificaiton != original_classification:
                        print(f'CF found for {solution_name}')
                    else:
                        print(f'CF not found for {solution_name}')
            except Exception as error:
                print(f'Error for {generator_name}')
                print(f'Error: {error}')
                if generator_name not in ['cfnow_greedy', 'cfnow_random']:
                    solution_name = generator_name
                    cf_classificaiton, exp_time, segments_in_explanation = None, None, None
                    # Save solution
                    pd.DataFrame([{
                        'image_hash': img_hash,
                        'original_classification': original_classification,
                        'cf_classification': cf_classificaiton,
                        'time': exp_time,
                        'segments_in_explanation': segments_in_explanation,
                        'generator': solution_name
                    }]).to_pickle(f'{SCRIPT_DIR}/Results/{img_hash}_{solution_name}.pkl')
                else:
                    for cfnow_type in ['optimized', 'not_optimized']:
                        solution_name = generator_name + '_' + cfnow_type
                        cf_classificaiton, exp_time, segments_in_explanation = None, None, None
                        # Save solution
                        pd.DataFrame([{
                            'image_hash': img_hash,
                            'original_classification': original_classification,
                            'cf_classification': cf_classificaiton,
                            'time': exp_time,
                            'segments_in_explanation': segments_in_explanation,
                            'generator': solution_name
                        }]).to_pickle(f'{SCRIPT_DIR}/Results/{img_hash}_{solution_name}.pkl')
                continue
