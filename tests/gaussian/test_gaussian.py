from src.gaussian import GaussianNB
from tests.config import dataset_classification, check_fit_predict  # noqa: F401


def test_gaussian_nb(dataset_classification):
    x, y = dataset_classification
    model = GaussianNB()
    check_fit_predict(model, x, y)
