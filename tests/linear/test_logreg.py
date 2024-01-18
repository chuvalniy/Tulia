from src.linear import LogisticRegression
from tests.config import dataset_classification, check_fit_predict  # noqa: F401


def test_logistic_regression(dataset_classification):
    x, y = dataset_classification
    model = LogisticRegression()
    check_fit_predict(model, x, y)
