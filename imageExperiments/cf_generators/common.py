import os
import copy

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ImageCounterfactualGenerator:

    def __init__(self, img, classifier, original_prediction_idx, replace_img, imagenet_labels, generator_name,
                 img_hash, skip_found=True):
        self.img = img
        self.classifier = classifier
        self.original_prediction_idx = original_prediction_idx
        self.replace_img = replace_img
        self.imagenet_labels = imagenet_labels
        self.skip_experiment = False

        experiment_names = [f'{img_hash}_{generator_name}.pkl']
        if generator_name in ['cfnow_greedy', 'cfnow_random']:
            experiment_names = [
                f'{img_hash}_{generator_name}_optimized.pkl',
                f'{img_hash}_{generator_name}_not_optimized.pkl']

        if skip_found:
            found_exp = 0
            for exp_name in experiment_names:
                if os.path.exists(f'{SCRIPT_DIR}/../Results/{exp_name}'):
                    found_exp += 1
            if found_exp == len(experiment_names):
                self.skip_experiment = True

    def explain(self, **kwargs):
        raise NotImplementedError

    # For feature importance methods, using their ranking, find the minimum set to change the classification
    def _find_cf_by_segment_importance(self, segments, segments_ranking, perturbed_image):
        # Generate the perturbed image segments
        image_segments = []
        explanation = copy.deepcopy(self.img)
        for ns in range(len(segments_ranking)):
            segments_in_explanation = segments_ranking[0:ns + 1]

            for i in segments_in_explanation:
                explanation[segments == i] = perturbed_image[segments == i]
            image_segments.append(copy.deepcopy(explanation))

        # Classify perturbed segments
        classifications = np.array([np.argmax(c) for c in self.classifier.predict(np.array(image_segments))])

        # Get segments in explanation
        segments_in_explanation = []
        for c, sr in zip(classifications, segments_ranking):
            segments_in_explanation.append(sr)
            if c != self.original_prediction_idx:
                break

        # Generate explanation image
        explanation = copy.deepcopy(self.img)
        for i in segments_in_explanation:
            explanation[segments == i] = perturbed_image[segments == i]

        return explanation, segments_in_explanation

    def _seg_to_img(self, segment_indexes, segments):
        # Get's a segmentation code and transforms to image data

        mask_replace = np.isin(segments, segment_indexes).astype(float)
        mask_original = (mask_replace == 0).astype(float)

        return self.img * np.stack((mask_original, mask_original, mask_original), axis=-1) + self.replace_img * \
               np.stack((mask_replace, mask_replace, mask_replace), axis=-1)

