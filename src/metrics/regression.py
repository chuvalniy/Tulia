import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean-squared error.
    :param y_true: Target labels.
    :param y_pred: Target predictions.
    :return: Loss.
    """
    n_examples = len(y_true)

    error = 1 / (2 * n_examples) * np.sum(np.square(y_pred - y_true))
    return error


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean-absolute error.
    :param y_true: Target labels.
    :param y_pred: Target predictions.
    :return: Loss.
    """
    error = np.mean(np.abs(y_true - y_pred))
    return error
