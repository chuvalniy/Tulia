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
    :return: Log loss (float).
    """
    loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / len(y_true)
    return loss


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate hinge loss for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: Hinge loss (float).
    """
    y = np.where(y_true <= 0, -1, 1)
    loss = np.mean(np.maximum(0, 1 - y_pred * y))
    return loss


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall score for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: Recall score (float).
    """

    true_positives = np.sum(y_true * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))

    if true_positives + false_negatives == 0:
        return 0.0

    recall = true_positives / (false_negatives + true_positives)
    return recall


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: Precision score (float).
    """

    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)

    if true_positives + false_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    return precision


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score for predictions.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Target predictions (n_samples, ).
    :return: F1 score (float).
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate ROC curve values.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Predictions probability (n_samples, ).
    :return:
        True positive rates (n_thresholds, ),
        False positive rates (n_thresholds, ).
        Thresholds (n_thresholds, ).
    """
    unique_probabilities = np.unique(y_pred)
    unique_probabilities = np.append(unique_probabilities, [2])  # Include upper bound

    thresholds = -np.sort(-unique_probabilities)

    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        predictions = np.array(y_pred >= threshold).astype(int)

        # Calculate true positive rates.
        tp = np.sum((y_true == 1) & (predictions == 1))
        fn = np.sum((y_true == 1) & (predictions == 0))
        if tp == 0 and fn == 0:
            tpr[i] = 0.0
        else:
            tpr[i] = tp / (tp + fn)

        # Calculate false positive rates.
        tn = np.sum((y_true == 0) & (predictions == 0))
        fp = np.sum((y_true == 0) & (predictions == 1))
        if fp == 0 and tn == 0:
            fpr[i] = 0.0
        else:
            fpr[i] = fp / (tn + fp)

    return tpr, fpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate ROC-AUC score using trapezoidal rule.
    :param y_true: Target labels (n_samples, ).
    :param y_pred: Predictions probability (n_samples, ).
    :return: ROC-AUC score.
    """

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    shifted_tpr = np.roll(tpr, shift=1)
    shifted_tpr[0] = 0

    height = np.diff(fpr, n=1, prepend=0)
    auc = np.sum((tpr + shifted_tpr) / 2 * height)
    return auc.item()
