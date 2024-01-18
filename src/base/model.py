from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class Model(ABC):
    """
    Interface to define machine learning models.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train model based on input data.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict scores for input data.
        :param x: Test data.
        :return: Test scores.
        """
        pass

    @abstractmethod
    def _predict(self, x: np.ndarray) -> Union[np.ndarray, float, int]:
        """
        Calculate predictions for input data.
        :param x: Input data.
        :return: Predictions / Single prediction.
        """