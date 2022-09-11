import time

import numpy as np

from cfnow import cf_finder

from imageExperiments.cf_generators.common import ImageCounterfactualGenerator


class CFNOWGreedyGenerator(ImageCounterfactualGenerator):

    def explain(self):

        cf = cf_finder.find_image(self.img, self.classifier.predict, cf_strategy='greedy')

        # Get optimized explanation classification
        cfnow_prediction_optimized = self.classifier.predict(
            np.array([self._seg_to_img(cf.cf_segments, cf.segments)]))
        cfnow_classification_optimized = self.imagenet_labels[np.argmax(cfnow_prediction_optimized)]

        # Get not optimized explanation classification
        cfnow_prediction_not_optimized = self.classifier.predict(
            np.array([self._seg_to_img(cf.cf_not_optimized_segments, cf.segments)]))
        cfnow_classification_not_optimized = self.imagenet_labels[np.argmax(cfnow_prediction_not_optimized)]

        return [cfnow_classification_optimized, cf.time_cf, cf.cf_segments], \
               [cfnow_classification_not_optimized, cf.time_cf_not_optimized, cf.cf_not_optimized_segments]


class CFNOWRandomGenerator(ImageCounterfactualGenerator):

    def explain(self):

        cf = cf_finder.find_image(self.img, self.classifier.predict, cf_strategy='random')

        # Get optimized explanation classification
        cfnow_prediction_optimized = self.classifier.predict(
            np.array([self._seg_to_img(cf.cf_segments, cf.segments)]))
        cfnow_classification_optimized = self.imagenet_labels[np.argmax(cfnow_prediction_optimized)]

        # Get not optimized explanation classification
        cfnow_prediction_not_optimized = self.classifier.predict(
            np.array([self._seg_to_img(cf.cf_not_optimized_segments, cf.segments)]))
        cfnow_classification_not_optimized = self.imagenet_labels[np.argmax(cfnow_prediction_not_optimized)]

        return [cfnow_classification_optimized, cf.time_cf, cf.cf_segments], \
               [cfnow_classification_not_optimized, cf.time_cf_not_optimized, cf.cf_not_optimized_segments]