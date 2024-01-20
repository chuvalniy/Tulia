import numpy as np

from src.metrics import precision_score


def test_precision_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_precision = 1.0
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)


def test_precision_different_labels():
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])

    expected_precision = 0.0
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)


def test_precision_reversed_different_labels():
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_precision = 0.0
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)


def test_precision_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_precision = 0.0
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)


def test_precision_all_true():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_precision = 0.4
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)


def test_recall_varied_true():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 1])

    expected_precision = 0.5
    precision = precision_score(y_true, y_pred)

    assert np.isclose(expected_precision, precision, atol=1e-5, rtol=1e-5)
