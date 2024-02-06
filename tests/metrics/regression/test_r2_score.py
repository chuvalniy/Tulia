import numpy as np

from src.metrics import r2_score


def test_r2_identical_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    expected_r2 = 1.0
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)


def test_r2_shifted_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 6])

    expected_r2 = 0.5
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)


def test_r2_reversed_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])

    expected_r2 = -3.0
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)


def test_r2_large_numbers():
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([15, 25, 35, 45, 55])

    expected_r2 = 0.875
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)


def test_mse_decimal_numbers():
    y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

    expected_r2 = 0.5
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)


def test_mse_with_outliers():
    y_true = np.array([10, 20, 30, 40, 135])
    y_pred = np.array([15, 25, 35, 45, 55])

    expected_r2 = 1 - 6500 / 10180
    r2 = r2_score(y_true, y_pred)

    assert np.isclose(expected_r2, r2, atol=1e-5, rtol=1e-5)
