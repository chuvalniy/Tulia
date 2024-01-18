import numpy as np

from src.neighbors import KNearestClassifier, KNearestRegressor
from tests.config import dataset_classification, check_fit_predict, dataset_regression  # noqa: F401


def test_knn_classifier(dataset_classification):
    x, y = dataset_classification
    model = KNearestClassifier()
    check_fit_predict(model, x, y)


def test_knn_classifier_prediction(dataset_classification):
    """
    Check if labels are the same for training data when k_nearest is equal to 1.
    """
    x, y = dataset_classification

    model = KNearestClassifier(k_nearest=1)
    model.fit(x, y)

    # When k_nearest=1 and we predict on X_train, predictions should be the same as y_train.
    y_pred = model.predict(x)

    assert np.allclose(y_pred, y)


def test_knn_regressor(dataset_regression):
    x, y = dataset_regression
    model = KNearestRegressor()
    check_fit_predict(model, x, y)


def test_knn_regressor_prediction(dataset_regression):
    """
    Check predictions for training data when k_nearest is equal to 1.
    """
    x, y = dataset_regression

    model = KNearestRegressor(k_nearest=1)
    model.fit(x, y)

    # When k_nearest=1 and we predict on X_train, predictions should be the same as y_train
    y_pred = model.predict(x)

    assert np.allclose(y_pred, y, atol=1e-5)
