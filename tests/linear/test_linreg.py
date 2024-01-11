from src.linear import LinearRegression
from tests.base.config import dataset_regression, check_fit_predict  # noqa: F401


def test_linear_regression(dataset_regression):
    x, y = dataset_regression
    model = LinearRegression()
    check_fit_predict(model, x, y)
