import numpy as np

from src.metrics import mean_absolute_error


def test_mae_identical_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    expected_mae = 0.0
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)


def test_mae_shifted_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 6])

    expected_mae = 1.0
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)


def test_mae_reversed_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])

    expected_mae = 2.4
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)


def test_mae_large_numbers():
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([15, 25, 35, 45, 55])

    expected_mae = 5.0
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)


def test_mae_decimal_numbers():
    y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

    expected_mae = 0.1
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)


def test_mse_with_outliers():
    y_true = np.array([10, 20, 30, 40, 135])
    y_pred = np.array([15, 25, 35, 45, 55])

    expected_mae = 20.0
    mae = mean_absolute_error(y_true, y_pred)

    assert np.isclose(expected_mae, mae, atol=1e-5, rtol=1e-5)
