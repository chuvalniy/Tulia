import numpy as np
from typing import Union
from abc import abstractmethod
from src.base import Model
from collections import Counter


class _KNN(Model):
    """
    K-nearest neighbors abstract class.
    """

    def __init__(self, k_nearest: int = 3):
        """
        :param k_nearest: Number of neighbours.
        """

        self.k_nearest = k_nearest
        self._x = None
        self._y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Remember training data.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """

        self._x = x
        self._y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict k-nearest neighbors in vectorized form using Euclidian distance.
        :param x: Test data.
        :return: Test scores.
        """
        euclidian_distance = np.linalg.norm(self._x[:, np.newaxis] - x, axis=2)

        k_nearest = np.argsort(euclidian_distance, axis=0)[:self.k_nearest].transpose()  # Shape is (n_examples, k)
        predictions = np.array([self._calculate_predictions(x) for x in self._y[k_nearest]])
        return predictions

    @abstractmethod
    def _calculate_predictions(self, x: np.ndarray) -> Union[int, float]:
        """
        Calculate prediction for a single sample.
        :param x: Single data sample (k_nearest)
        :return:
        """
        pass


class KNearestClassifier(_KNN):
    """
    KNN for binary/multiclass classification.
    """

    def _calculate_predictions(self, x: np.ndarray) -> Union[int, float]:
        """
        Find most common neighbor for a data sample.
        :param x: Single data sample (k_nearest)
        :return: Most common class.
        """
        most_common = np.bincount(x).argmax()
        return most_common


class KNearestRegressor(_KNN):
    """
    KNN for regression.
    """

    def _calculate_predictions(self, x: np.ndarray) -> Union[int, float]:
        """
        Find the mean value of neighbors for a data sample.
        :param x: Single data sample (k_nearest)
        :return: Most common class.
        """
        prediction = np.mean(x)
        return prediction
