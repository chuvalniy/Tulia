import numpy as np

from src.linear import LinearRegression
from tests.base.config import dataset_regression  # noqa: F401


class TestLinearRegression:
    def test_fit(self, dataset_regression):
        """
        Test if 'fit' is actually reaches the end.
        :param dataset_regression: Two arrays of training and target data.
        :return:
        """
        model = LinearRegression()

        x, y = dataset_regression

        # Fit the model on the mock dataset
        assert model.theta is None
        model.fit(x, y)
        assert isinstance(model.theta, np.ndarray)
        assert model.theta.shape[0] == x.shape[1] + 1

    def test_predict(self, dataset_regression):
        """
        Test if 'predict' is actually reaches the end.
        :param dataset_regression: Two arrays of training and target data.
        :return:
        """
        model = LinearRegression()

        x, y = dataset_regression

        # Fit the model on the mock dataset
        model.fit(x, y)

        predictions = model.predict(x)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
