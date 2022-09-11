import time

import cv2
import numpy as np
from skimage.segmentation import quickshift

from imageExperiments.cf_generators.common import ImageCounterfactualGenerator


class OcclusionGenerator(ImageCounterfactualGenerator):

    def explain(self):
        initial_time = time.time()

        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
        segments = quickshift(self.img, **params_seg)

        # Create the perturbed image
        perturbed_image = cv2.GaussianBlur(self.img, (31, 31), 0)

        result = self.classifier.predict(self.img[np.newaxis, ...])
        c = np.argmax(result)
        p = result[0, c]
        P = np.array([])  # corresponding scores for original class

        perturbed_image_oc = np.zeros((224, 224, 3))
        perturbed_image_oc[:, :, 0] = np.mean(self.img[:, :, 0])
        perturbed_image_oc[:, :, 1] = np.mean(self.img[:, :, 1])
        perturbed_image_oc[:, :, 2] = np.mean(self.img[:, :, 2])

        applied_segments = np.unique(segments)

        for j in applied_segments:
            test_image = self.img.copy()
            test_image[segments == j] = perturbed_image_oc[segments == j]

            result = self.classifier.predict(test_image[np.newaxis, ...])
            p_new = result[0, c]
            P = np.append(P, p - p_new)

        P_sorted = np.argsort(-P)
        segments_ranking = applied_segments[P_sorted]

        explanation, segments_in_explanation = self._find_cf_by_segment_importance(
            segments, segments_ranking, perturbed_image)

        exp_time = time.time() - initial_time

        oc_prediction = self.classifier.predict(
            np.array([self._seg_to_img(segments_in_explanation, segments)]))
        oc_classification = self.imagenet_labels[np.argmax(oc_prediction)]

        return oc_classification, exp_time, segments_in_explanation
