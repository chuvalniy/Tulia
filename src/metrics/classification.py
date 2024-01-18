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
