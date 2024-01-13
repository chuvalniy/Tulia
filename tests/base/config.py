import numpy as np
import pytest


@pytest.fixture(scope='module')
def dataset_regression() -> (np.ndarray, np.ndarray):
    # Create a feature matrix with 100 rows and 5 columns
    X = np.array([[-0.68372786, -0.12289023, 0.90085595],
                  [-0.38405435, 1.13376944, -0.3224172],
                  [0.86540763, -2.3015387, -1.07296862],
                  [-0.26788808, 0.53035547, -0.93576943],
                  [-0.17242821, -0.87785842, -1.09989127],
                  [-0.7612069, 0.3190391, 1.74481176],
                  [0.90159072, 0.50249434, 1.14472371],
                  [-0.61175641, -0.52817175, 1.62434536],
                  [0.58281521, -1.10061918, 0.04221375],
                  [1.46210794, -2.06014071, -0.24937038]])

    # Create a target vector with 100 random values
    y = np.array([16.79599588, -12.07640944, -24.08199932, -29.38530983,
                  -35.59436225, 40.93139665, 45.63875939, 37.7671644,
                  6.66150601, 7.84545048])

    return X, y


@pytest.fixture(scope='module')
def dataset_classification() -> (np.ndarray, np.ndarray):
    # Create a feature matrix with 100 rows and 5 columns
    X = np.array([[1.65980218, -1.14651383, 1.44634283],
                  [-0.67124613, -2.85961623, -1.52637437],
                  [-0.0126646, 1.49509867, -3.42524143],
                  [-0.6871727, -2.00875146, -0.60483688],
                  [-0.84520564, -1.52320683, -1.60290743],
                  [-0.19183555, 1.72073855, 1.23169963],
                  [0.2344157, 0.70896364, -4.42287433],
                  [0.74204416, -0.22942496, 1.37496472],
                  [-0.88762896, 0.55132541, 0.78712117],
                  [-1.11731035, 1.03584131, -1.55118469]])
    # Create a target vector with 100 binary labels
    y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])

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
