from typing import Union

import numpy as np

from src.base import Model


# 1. Ordered Target encoding
# 2. Boostrap data
# 3. Symmetric tree
# 4.


class CatBoostClassifier(Model):
    """
    CatBoost for classification tasks.
    """

    def __init__(
            self,
            learning_rate: float = 3e-1,
            n_steps: int = 100,
            max_depth: int = 3,
            cat_features: list = None
    ):
        pass


    def fit(self, x: np.ndarray, y: np.ndarray):
        pass
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def _predict(self, x: np.ndarray) -> Union[np.ndarray, float, int]:
        pass

    def _encode_cat_features(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cat_feature_idxs = []

        encoded_features = []
        for idx in cat_feature_idxs:
            encoded_feature = []
            option_count = {}
            total_count = {}
            for i, x_sample in enumerate(x[:, idx]):
                ctr = (option_count.get(x_sample, 0) + 0.05) / (total_count.get(x_sample, 0) + 1)
                encoded_feature.append(ctr)

                if y[i] == 1:
                    option_count[x_sample] = option_count.get(x_sample, 0) + 1
                total_count[x_sample] = total_count.get(x_sample, 0) + 1

            encoded_features.append(np.array(encoded_feature))

        x[:, cat_feature_idxs] = encoded_features
        return x
