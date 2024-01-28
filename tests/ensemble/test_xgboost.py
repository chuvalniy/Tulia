from src.ensemble import XGBRegressionTree, XGBRegressor
from tests.config import dataset_regression, dataset_classification, check_fit_predict  # noqa: F401


def test_xgboost_regression_tree(dataset_regression):
    x, y = dataset_regression
    model = XGBRegressionTree()
    check_fit_predict(model, x, y)


def test_xgboost_regression(dataset_regression):
    x, y = dataset_regression
    model = XGBRegressor()
    check_fit_predict(model, x, y)
