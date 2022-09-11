import time

import cv2
import shap
import numpy as np
from skimage.segmentation import quickshift

from imageExperiments.cf_generators.common import ImageCounterfactualGenerator


class SHAPCGenerator(ImageCounterfactualGenerator):

    def explain(self):

        initial_time = time.time()

        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
        segments = quickshift(self.img, **params_seg)

        # Create the perturbed image
        perturbed_image = cv2.GaussianBlur(self.img, (31, 31), 0)

        # define a function that depends on a binary mask representing if an image region is hidden
        def mask_image(zs, segmentation, img, background=None):
            if background is None:
                background = img.mean((0, 1))
            out = np.zeros((zs.shape[0], img.shape[0], img.shape[1], img.shape[2]))
            for i in range(zs.shape[0]):
                out[i, :, :, :] = img
                for j in range(zs.shape[1]):
                    if zs[i, j] == 0:
                        out[i][segmentation == j, :] = background
            return out

        def f(z):
            return self.classifier.predict(mask_image(z, segments, self.img))

        # use Kernel SHAP to explain the network's predictions
        n_segments = len(np.unique(segments))
        explainer = shap.KernelExplainer(f, np.zeros((1, n_segments)))
        shap_values = explainer.shap_values(np.ones((1, n_segments)), nsamples=1000)

        relevant_shap = shap_values[np.argmax(self.classifier.predict(self.img[np.newaxis, ...])[0], axis=-1)]
        ranked_segments = np.argsort(-relevant_shap)[0]

        explanation, segments_in_explanation = self._find_cf_by_segment_importance(
            segments, ranked_segments, perturbed_image)

        exp_time = time.time() - initial_time

        shap_prediction = self.classifier.predict(
            np.array([self._seg_to_img(segments_in_explanation, segments)]))
        shap_classification = self.imagenet_labels[np.argmax(shap_prediction)]

        return shap_classification, exp_time, segments_in_explanation
