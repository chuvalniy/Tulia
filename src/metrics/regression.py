import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean-squared error.
    :param y_true: Target labels (n_examples, ).
    :param y_pred: Target predictions (n_examples, ).
    :return: Loss.
    """
    n_examples = len(y_true)

    error = 1 / (2 * n_examples) * np.sum(np.square(y_pred - y_true))
    return error


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean-absolute error.
    :param y_true: Target labels (n_examples, ).
    :param y_pred: Target predictions (n_examples, ).
    :return: Loss.
    """
    error = np.mean(np.abs(y_true - y_pred))
    return error


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared.
    :param y_true: Target labels (n_examples, ).
    :param y_pred: Target predictions (n_examples, ).
    :return: R-squared score.
    """

    tss = np.sum((y_true - np.mean(y_true))**2)
    rss = np.sum((y_true - y_pred)**2)

    r_squared = 1 - rss / tss
    return r_squared
