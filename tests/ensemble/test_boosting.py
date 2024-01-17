from src.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from tests.base.config import dataset_regression, dataset_classification, check_fit_predict  # noqa: F401


def test_gradient_boosting_regression(dataset_regression):
    x, y = dataset_regression
    model = GradientBoostingRegressor(learning_rate=5e-2)
    check_fit_predict(model, x, y)


def test_gradient_boosting_classification(dataset_classification):
    x, y = dataset_classification
    model = GradientBoostingClassifier()
    check_fit_predict(model, x, y)
