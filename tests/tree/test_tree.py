from src.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tests.config import dataset_regression, dataset_classification, check_fit_predict  # noqa: F401


def test_decision_tree_regression(dataset_regression):
    x, y = dataset_regression
    model = DecisionTreeRegressor()
    check_fit_predict(model, x, y)


def test_decision_tree_classification(dataset_classification):
    x, y = dataset_classification
    model = DecisionTreeClassifier()
    check_fit_predict(model, x, y)
