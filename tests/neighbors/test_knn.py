from src.neighbors import KNN
from tests.base.config import dataset_classification, check_fit_predict  # noqa: F401


def test_knn(dataset_classification):
    x, y = dataset_classification
    model = KNN()
    check_fit_predict(model, x, y)
