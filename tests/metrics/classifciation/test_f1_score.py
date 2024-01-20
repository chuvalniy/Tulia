import numpy as np

from src.metrics import f1_score


def test_f1_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_f1 = 1.0
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)


def test_f1_different_labels():
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])

    expected_f1 = 0.0
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)


def test_f1_reversed_different_labels():
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_f1 = 0.0
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)


def test_f1_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_f1 = 0.0
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)


def test_f1_all_true():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_f1 = 0.57142857142
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)


def test_f1_varied_true():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 1])

    expected_f1 = 0.57142857142
    f1 = f1_score(y_true, y_pred)

    assert np.isclose(expected_f1, f1, atol=1e-5, rtol=1e-5)
