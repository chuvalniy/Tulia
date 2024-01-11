from src.ensemble import GradientBoostingRegressor
from tests.base.config import dataset_regression, check_fit_predict  # noqa: F401


def test_gradient_boosting_regression(dataset_regression):
    x, y = dataset_regression
    model = GradientBoostingRegressor()
    check_fit_predict(model, x, y)
