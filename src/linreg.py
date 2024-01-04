import numpy as np

from src.model import Model


class LinearRegression(Model):
    """
    Vanilla Linear Regression.
    """

    def __init__(self, learning_rate: float = 1e-3, n_steps: int = 1000):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        """
        self.theta = None
        self.learning_rate = learning_rate

        self.error = 0.0
        self.n_steps = n_steps

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train linear regression using Mean-Squared Error.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """
        n_examples, n_features = x.shape

        # Consider bias by adding one extra parameter.
        self.theta = np.random.randn(n_features + 1,)

        bias_term = np.ones((n_examples, 1))
        x_copy = np.concatenate((x, bias_term), axis=1)

        # Training process.
        for _ in range(self.n_steps):
            self.error = 1 / (n_examples * 2) * np.sum((x_copy @ self.theta - y) ** 2)
            dtheta = 1 / n_examples * np.sum((x_copy @ self.theta - y)[:, np.newaxis] * x_copy, axis=0)
            self.theta = self.theta - self.learning_rate * dtheta

    def predict(self, x: np.ndarray):
        """
        Predict target feature using theta parameters.
        :param x: Test data.
        :return: Test predictions.
        """
        n_examples, n_features = x.shape

        # Add bias term for test data.
        bias_term = np.ones((n_examples, 1))
        x_copy = np.concatenate((x, bias_term), axis=1)

        return x_copy @ self.theta
