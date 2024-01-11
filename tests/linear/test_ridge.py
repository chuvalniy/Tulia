from src.linear import RidgeRegression
from tests.base.config import dataset_regression, check_fit_predict  # noqa: F401


def test_ridge_regression(dataset_regression):
    x, y = dataset_regression
    model = RidgeRegression()
    check_fit_predict(model, x, y)
