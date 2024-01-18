import numpy as np

from src.metrics import mean_squared_error


def test_mse_identical_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    expected_mse = 0.0
    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(expected_mse, mse, atol=1e-5, rtol=1e-5)


def test_mse_shifted_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 6])

    expected_mse = 0.5
    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(expected_mse, mse, atol=1e-5, rtol=1e-5)


def test_mse_reversed_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])

    expected_mse = 4.0
    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(expected_mse, mse, atol=1e-5, rtol=1e-5)


def test_mse_large_numbers():
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([15, 25, 35, 45, 55])

    expected_mse = 12.5
    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(expected_mse, mse, atol=1e-5, rtol=1e-5)


def test_mse_decimal_numbers():
    y_true = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

    expected_mse = 0.005
    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(expected_mse, mse, atol=1e-5, rtol=1e-5)