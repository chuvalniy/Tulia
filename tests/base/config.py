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

    return X, y

def check_fit_predict(model, x: np.ndarray, y: np.ndarray):
    """
    Test to check if model's 'fit' and 'predict' methods are actually working.
    :param model: Machine learning model.
    :param x: Training data.
    :param y: Targets.
    :return:
    """

    # Fit the model on the mock dataset
    model.fit(x, y)
    preds = model.predict(x)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
