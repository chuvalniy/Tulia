from src.ensemble import RandomForestClassifier, RandomForestRegressor
from tests.config import dataset_regression, dataset_classification, check_fit_predict  # noqa: F401


def test_random_forest_regression(dataset_regression):
    x, y = dataset_regression
    model = RandomForestRegressor(n_trees=10)
    check_fit_predict(model, x, y)


def test_random_forest_classification(dataset_classification):
    x, y = dataset_classification
    model = RandomForestClassifier(n_trees=10)
    check_fit_predict(model, x, y)
