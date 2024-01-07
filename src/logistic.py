import numpy as np

from src.model import Model


class LogisticRegression(Model):
    """
    Logistic Regression (Classification)
    Currently implemented only binary classification.
    """

    def __init__(self, learning_rate: float = 1e-3, eps: float = 1e-5, n_steps: int = 1000, tol: float = 1e-5):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        :param eps: Small number to prevent log of 0.
        :param tol: Tolerance value to terminate training process if function converges.
        """
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_steps = n_steps
        self.tol = tol

        self.theta = None  # Model parameters.
        self.logloss = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train logistic regression.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """
        n_examples, n_features = x.shape

        # Consider bias by adding one extra parameter.
        self.theta = np.random.randn(n_features + 1)

        bias_term = np.ones((n_examples, 1))
        x_copy = np.concatenate((x, bias_term), axis=1)

        for _ in range(self.n_steps):
            # Calculate probability for each example (via sigmoid function) and binary cross-entropy (logloss).
            logits = 1 / (1 + np.exp(-x_copy @ self.theta))
            self.logloss = -np.sum(y * np.log(logits + self.eps) + (1 - y) * np.log(1 - logits + self.eps)) / n_examples

            # Calculate derivatives step-by-step using backpropagation.
            dlogits = (logits - y) / n_examples
            dtheta = np.dot(x_copy.T, dlogits)

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

        logits = 1 / (1 + np.exp(-x_copy @ self.theta))
        predictions = (logits >= 0.5).astype(int)  # Convert from float numbers to discrete classes.

        return predictions
