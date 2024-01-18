import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: Accuracy score (float).
    """
    n_samples = len(y_true)

    accuracy = np.sum(y_true == y_pred) / n_samples
    return accuracy


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate log loss for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: Accuracy score (float).
    """
    loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / len(y_true)
    return loss
