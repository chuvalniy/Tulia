import numpy as np

from src.metrics import roc_curve


def test_roc_curve_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_tpr = np.array([0, 1, 1])
    expected_fpr = np.array([0, 0, 1])

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    assert np.allclose(expected_tpr, tpr, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_fpr, fpr, atol=1e-5, rtol=1e-5)


def test_roc_curve_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_tpr = np.array([0, 0, 1])
    expected_fpr = np.array([0, 1, 1])

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    assert np.allclose(expected_tpr, tpr, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_fpr, fpr, atol=1e-5, rtol=1e-5)


def test_roc_curve_all_true():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_tpr = np.array([0, 1])
    expected_fpr = np.array([0, 1])

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    assert np.allclose(expected_tpr, tpr, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_fpr, fpr, atol=1e-5, rtol=1e-5)


def test_roc_curve_equal_prob():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    expected_tpr = np.array([0, 1])
    expected_fpr = np.array([0, 1])

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    assert np.allclose(expected_tpr, tpr, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_fpr, fpr, atol=1e-5, rtol=1e-5)


def test_roc_curve_close_to_target():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.1, 0.1, 0.9])

    expected_tpr = np.array([0, 1, 1])
    expected_fpr = np.array([0, 0, 1])

    tpr, fpr, _ = roc_curve(y_true, y_pred)

    assert np.allclose(expected_tpr, tpr, atol=1e-5, rtol=1e-5)
    assert np.allclose(expected_fpr, fpr, atol=1e-5, rtol=1e-5)
