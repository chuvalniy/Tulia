from src.linear import LassoRegression
from tests.base.config import dataset_regression, check_fit_predict  # noqa: F401


def test_lasso_regression(dataset_regression):
    x, y = dataset_regression
    model = LassoRegression()
    check_fit_predict(model, x, y)
