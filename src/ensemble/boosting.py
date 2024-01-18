from abc import abstractmethod

import numpy as np

from src.base import Model, ClassifierMixin, RegressorMixin
from src.tree import DecisionTreeRegressor


class _GradientBoosting(Model):
    def __init__(
            self,
            learning_rate: float = 1e-2,
            n_steps: int = 100,
            max_depth: int = 3,
            min_samples_split: int = 2,
            max_features: float = 1.0
    ):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        :param max_depth: Maximum depth of a decision tree.
        :param min_samples_split: Minimum number of samples to split data into right and left nodes.
        :param max_features: Percentage of features to use for training.
        """

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.trees = None
        self.constant_prediction = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train gradient boosting.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """

        # Initial prediction
        self.constant_prediction = np.mean(y)

        prediction = self.constant_prediction
        self.trees = []
        for _ in range(self.n_steps):
            residuals = self._calculate_loss_gradient(y, prediction)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )

            tree.fit(x, residuals)
            prediction = prediction + self.learning_rate * tree.predict(x)

            self.trees.append(tree)

    @abstractmethod
    def _calculate_loss_gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Find gradient for the loss function.
        :param y: Targets.
        :param predictions. Prediction to targets.
        :return: Initial predictions.
        """
        pass

    @abstractmethod
    def _calculate_predictions(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for input data.
        :param x: Input data.
        :return: Predictions.
        """

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target feature using pretrained boosting trees.
        :param x: Test data.
        :return: Test predictions.
        """
        predictions = self._calculate_predictions(x)
        return predictions


class GradientBoostingRegressor(_GradientBoosting, RegressorMixin):
    """
    Gradient Boosting for regression
    """

    def _calculate_loss_gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of mean-squared error loss.
        :param predictions: Target predictions.
        :param y: Targets.
        :return: Gradient of loss function with respect to predictions.
        """
        return y - predictions

    def _calculate_predictions(self, x: np.ndarray) -> np.ndarray:
        n_samples, _ = x.shape

        predictions = np.ones(n_samples) * self.constant_prediction
        for tree in self.trees:
            predictions = predictions + self.learning_rate * tree.predict(x)

        return predictions


class GradientBoostingClassifier(_GradientBoosting, ClassifierMixin):
    """
    Gradient Boosting for the classification.
    Uses cross-entropy as loss.
    """

    def _calculate_loss_gradient(self, y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate cross-entropy gradient.
        :param y: Targets.
        :return: Gradient of loss function with respect to predictions.
        """
        return y - GradientBoostingClassifier.sigmoid(predictions)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Makes input values to be in (0, 1) range.
        :param x: Input array.
        :return: Output array of the same shape as an input array.
        """
        return 1 / (1 + np.exp(-x))

    def _calculate_predictions(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate targets using prediction probability.
        :param x: Input array.
        :return: Predictions.
        """
        predictions_proba = self.predict_proba(x)
        predictions = np.where(predictions_proba >= 0.5, 1, 0)
        return predictions

    def predict_proba(self, x):
        """
        Predict label using sigmoid function.
        :param x: Input array.
        :return: Predictions.
        """
        n_samples, _ = x.shape

        predictions = np.ones(n_samples) * self.constant_prediction
        for tree in self.trees:
            predictions = predictions + self.learning_rate * tree.predict(x)

        return GradientBoostingClassifier.sigmoid(predictions)

