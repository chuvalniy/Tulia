from src.neighbors import KNearestClassifier
from tests.base.config import dataset_classification, check_fit_predict  # noqa: F401


def test_knn_classifier(dataset_classification):
    x, y = dataset_classification
    model = KNearestClassifier()
    check_fit_predict(model, x, y)
