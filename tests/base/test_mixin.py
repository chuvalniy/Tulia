from typing import Union

import numpy as np

from src.base import Model, ClassifierMixin, RegressorMixin, is_regressor, is_classifier


class BaseModel(Model):

    def _predict(self, x: np.ndarray) -> Union[np.ndarray, float]:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class MockRegressionModel(BaseModel, RegressorMixin):
    pass


class MockClassifierModel(BaseModel, ClassifierMixin):
    pass


def test_if_model_is_classifier():
    model = MockClassifierModel()

    assert is_classifier(model)
    assert not is_regressor(model)


def test_if_model_is_regressor():
    model = MockRegressionModel()

    assert is_regressor(model)
    assert not is_classifier(model)
