import numpy as np

from src.model import Model


class RidgeRegression(Model):
    """
    Ridge Regression (L2)
    """

    def __init__(self, learning_rate: float = 1e-3, alpha: float = 1.0, n_steps: int = 1000, tol: float = 1e-5):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        :param alpha: Regularization strength for L1.
        :param tol: Tolerance value to terminate training process if function converges.
        """
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_steps = n_steps
        self.tol = tol

        self.error = 0.0
        self.theta = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train linear regression using Mean-Squared Error with L2 (Ridge) regularization.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """
        n_examples, n_features = x.shape

        # Consider bias by adding one extra parameter.
        self.theta = np.random.randn(n_features + 1)

        bias_term = np.ones((n_examples, 1))
        x_copy = np.concatenate((x, bias_term), axis=1)

        prev_error = None
        for _ in range(self.n_steps):
            # Calculate error function using mean-squared method with L2 regularization.
            mean_squared = 1 / (2 * n_examples) * np.sum((x_copy @ self.theta - y) ** 2)
            regularization = self.alpha * np.sum(self.theta ** 2)
            self.error = mean_squared + regularization

            # Terminate training process if function converges.
            if prev_error and np.isclose(self.error, prev_error, atol=self.tol, rtol=self.tol):
                return
            prev_error = self.error

            # Find derivatives for both least squared and regularization terms from the error function.
            dtheta_mean_squared = 1 / n_examples * np.sum((x_copy @ self.theta - y)[:, np.newaxis] * x_copy, axis=0)
            dtheta_regularization = self.alpha * 2 * np.sum(self.theta)
            dtheta = dtheta_mean_squared + dtheta_regularization

            # Update theta value by making gradient descent step.
            self.theta = self.theta - dtheta * self.learning_rate

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target feature using theta parameters.
        :param x: Test data.
        :return: Test predictions.
        """
        n_examples, n_features = x.shape

        bias_term = np.ones((n_examples, 1))
        x_copy = np.concatenate((x, bias_term), axis=1)

        return x_copy @ self.theta
