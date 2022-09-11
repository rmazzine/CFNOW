import time

import cv2
import numpy as np
from skimage.segmentation import quickshift

from imageExperiments.cf_generators.common import ImageCounterfactualGenerator


class SEDCGenerator(ImageCounterfactualGenerator):

    def explain(self):
        max_time = 600
        mode = 'blur'
        init_time = time.time()

        params_seg = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
        segments = quickshift(self.img, **params_seg)

        result = self.classifier.predict(self.img[np.newaxis, ...])

        c = np.argmax(result)
        p = result[0, c]
        R = []  # list of explanations
        I = []  # corresponding perturbed images
        C = []  # corresponding new classes
        P = []  # corresponding scores for target class
        sets_to_expand_on = []
        P_sets_to_expand_on = np.array([])

        if mode == 'mean':
            perturbed_image = np.zeros((224, 224, 3))
            perturbed_image[:, :, 0] = np.mean(self.img[:, :, 0])
            perturbed_image[:, :, 1] = np.mean(self.img[:, :, 1])
            perturbed_image[:, :, 2] = np.mean(self.img[:, :, 2])
        elif mode == 'blur':
            perturbed_image = cv2.GaussianBlur(self.img, (31, 31), 0)
        elif mode == 'random':
            perturbed_image = np.random.random((224, 224, 3))
        elif mode == 'inpaint':
            perturbed_image = np.zeros((224, 224, 3))
            for j in np.unique(segments):
                image_absolute = (self.img * 255).astype('uint8')
                mask = np.full([image_absolute.shape[0], image_absolute.shape[1]], 0)
                mask[segments == j] = 255
                mask = mask.astype('uint8')
                image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
                perturbed_image[segments == j] = image_segment_inpainted[segments == j] / 255.0

        cf_candidates = []
        for j in np.unique(segments):
            test_image = self.img.copy()
            test_image[segments == j] = perturbed_image[segments == j]

            cf_candidates.append(test_image[np.newaxis, ...][0])

        cf_candidates = np.array(cf_candidates)

        results = self.classifier.predict(cf_candidates)
        c_new_list = np.argmax(results, axis=1)
        p_new_list = results[:, c]

        if len(set(c_new_list) - set([c])) > 0:
            R = [[x] for x in np.where(c_new_list != c)[0]]

            target_class_idxs = np.array(R).reshape(1, -1)[0]

            I = cf_candidates[target_class_idxs]
            C = c_new_list[target_class_idxs]
            P = p_new_list[target_class_idxs]

        sets_to_expand_on = [[x] for x in np.where(c_new_list == c)[0]]
        P_sets_to_expand_on = p - results[np.where(c_new_list == c)[0], c]

        combo_set = [0]

        while len(R) == 0 and len(combo_set) > 0 and max_time > time.time() - init_time:

            combo = np.argmax(P_sets_to_expand_on)
            combo_set = []
            for j in np.unique(segments):
                if j not in sets_to_expand_on[combo]:
                    combo_set.append(np.append(sets_to_expand_on[combo], j))

            # Make sure to not go back to previous node
            del sets_to_expand_on[combo]
            P_sets_to_expand_on = np.delete(P_sets_to_expand_on, combo)

            cf_candidates = []

            for cs in combo_set:
                test_image = self.img.copy()
                for k in cs:
                    test_image[segments == k] = perturbed_image[segments == k]

                cf_candidates.append(test_image)
            cf_candidates = np.array(cf_candidates)

            results = self.classifier.predict(cf_candidates)
            c_new_list = np.argmax(results, axis=1)
            p_new_list = results[:, c]

            if len(set(c_new_list) - set([c])) > 0:
                selected_idx = np.where(c_new_list != c)[0]

                R = np.array(combo_set)[selected_idx].tolist()
                I = cf_candidates[selected_idx]
                C = c_new_list[selected_idx]
                P = p_new_list[selected_idx]

            sets_to_expand_on += np.array(combo_set)[np.where(c_new_list == c)[0]].tolist()
            P_sets_to_expand_on = np.append(P_sets_to_expand_on, p - results[np.where(c_new_list == c)[0], c])

        # Select best explanation: highest score reduction

        if len(R) > 0:
            best_explanation = np.argmax(p - P)
            segments_in_explanation = R[best_explanation]
            explanation = np.full([self.img.shape[0], self.img.shape[1], self.img.shape[2]], 0 / 255.0)
            for i in R[best_explanation]:
                explanation[segments == i] = self.img[segments == i]
            perturbation = I[best_explanation]
            new_class = C[best_explanation]

            sedc_classification = self.imagenet_labels[new_class]
            return sedc_classification, time.time() - init_time, segments_in_explanation

        sedc_classification = self.imagenet_labels[c]
        print('No CF found on the requested parameters')
        return sedc_classification, time.time() - init_time, None
