import numpy as np

from src.base import ClassifierMixin
from src.metrics import hinge_loss
from .linear import _Linear


class SoftSVC(_Linear, ClassifierMixin):
    """
    Soft margin Support Vector Machines (classification).
    """

    def __init__(self, learning_rate: float = 1e-2, n_steps: int = 1000, alpha: float = 1e-3, tol: float = 1e-5):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        :param alpha: Regularization strength for L2.
        :param tol: Tolerance value to terminate a training process if function converges.
        """
        super().__init__(learning_rate=learning_rate, n_steps=n_steps, tol=tol)

        self.alpha = alpha
        self._scores = None

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        :param x: Input data.
        :return: Predictions.
        """
        logits = x @ self.theta

        predictions = np.where(np.sign(logits) == -1, 0, 1)  # Transform labels to be either 0 or 1.
        return predictions

    def _calculate_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate hinge loss with L2 regularization.
        :param x: Training data.
        :param y: Targets.
        :return: Mean-squared error with L2 regularization.
        """
        self._scores = x @ self.theta
        hingeloss = hinge_loss(self._scores, y)
        regularization = self.alpha * np.sum(self.theta ** 2)

        loss = hingeloss + regularization
        return loss

    def _calculate_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find gradient of loss function with respect to theta using backpropagation.
        :param x: Training data.
        :param y: Targets.
        :return: Gradient with respect to theta.
        """
        n_examples, _ = x.shape

        dtheta = 2 * self.alpha * self.theta  # Derivative for regularization term.
        dscores = np.where(self._scores > 0, 0, -y) / n_examples  # Derivative of a hinge loss with respect to scores.
        dtheta += np.dot(x.transpose(), dscores)  # Derivative of scores with respect to theta.

        return dtheta
