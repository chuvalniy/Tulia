import numpy as np
import pytest


@pytest.fixture(scope='module')
def dataset_regression() -> (np.ndarray, np.ndarray):
    # Create a feature matrix with 100 rows and 5 columns
    X = np.zeros((100, 5))
    for i in range(100):
        X[i, :] = np.array([i, i + 1, i + 2, i + 3, i + 4])

    # Create a target vector with 100 random values
    y = np.random.rand(100)

    return X, y


@pytest.fixture(scope='module')
def dataset_classification() -> (np.ndarray, np.ndarray):
    # Create a feature matrix with 100 rows and 5 columns
    X = np.zeros((100, 5))
    for i in range(100):
        X[i, :] = np.array([i, i + 1, i + 2, i + 3, i + 4])

    # Create a target vector with 100 binary labels
    y = np.random.randint(0, 2, size=100)
