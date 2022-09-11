import time

import cv2
import numpy as np
from lime import lime_image
from skimage.segmentation import quickshift

from imageExperiments.cf_generators.common import ImageCounterfactualGenerator


class LIMECGenerator(ImageCounterfactualGenerator):

    def explain(self):

        # Start timer
        initial_time = time.time()

        # Set segmentation parameters
        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}

        # Create the perturbed image
        perturbed_image = cv2.GaussianBlur(self.img, (31, 31), 0)

        # Create explainer and explain
        explainer = lime_image.LimeImageExplainer()
        explanation_lime = explainer.explain_instance(self.img, self.classifier.predict, top_labels=2, hide_color=0,
                                                      num_samples=1000, segmentation_fn=lambda x: quickshift(x, **params_seg),
                                                      random_seed=42)

        # Get only positive explanations that explain the image
        positive_explanations = [exp_s[0] for exp_s in explanation_lime.local_exp[self.original_prediction_idx] if exp_s[1] > 0]
        segments = explanation_lime.segments

        explanation, segments_in_explanation = self._find_cf_by_segment_importance(
            segments, positive_explanations, perturbed_image)

        exp_time = time.time() - initial_time

        lime_prediction = self.classifier.predict(
            np.array([self._seg_to_img(segments_in_explanation, segments)]))
        lime_classification = self.imagenet_labels[np.argmax(np.array(lime_prediction))]

        return lime_classification, exp_time, segments_in_explanation
