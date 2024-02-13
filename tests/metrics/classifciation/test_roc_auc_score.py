import numpy as np

from src.metrics import roc_auc_score


def test_roc_auc_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_roc_auc = 1.0
    roc_auc = roc_auc_score(y_true, y_pred)

    assert np.isclose(expected_roc_auc, roc_auc, atol=1e-5, rtol=1e-5)


def test_roc_auc_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_roc_auc = 0.0
    roc_auc = roc_auc_score(y_true, y_pred)

    assert np.isclose(expected_roc_auc, roc_auc, atol=1e-5, rtol=1e-5)


def test_roc_auc_all_true():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_roc_auc = 0.5
    roc_auc = roc_auc_score(y_true, y_pred)

    np.isclose(expected_roc_auc, roc_auc, atol=1e-5, rtol=1e-5)


def test_roc_auc_equal_prob():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    expected_roc_auc = 0.5
    roc_auc = roc_auc_score(y_true, y_pred)

    np.isclose(expected_roc_auc, roc_auc, atol=1e-5, rtol=1e-5)


def test_roc_auc_close_to_target():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.1, 0.1, 0.9])

    expected_roc_auc = 1.0
    roc_auc = roc_auc_score(y_true, y_pred)

    np.isclose(expected_roc_auc, roc_auc, atol=1e-5, rtol=1e-5)
