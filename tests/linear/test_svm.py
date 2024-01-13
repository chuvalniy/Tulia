from src.linear import SoftSVC
from tests.base.config import dataset_classification, check_fit_predict  # noqa: F401


def test_soft_svm_classifier(dataset_classification):
    x, y = dataset_classification
    model = SoftSVC()
    check_fit_predict(model, x, y)
