from .mixin import ClassifierMixin, RegressorMixin
from .model import Model


def is_classifier(model: Model) -> bool:
    """
    Check if estimator is classifier.
    :param model: Machine learning model.
    :return: True if classifier else False.
    """
    return isinstance(model, ClassifierMixin)


def is_regressor(model: Model) -> bool:
    """
    Check if estimator is regressor.
    :param model: Machine learning model.
    :return: True if regressor else False.
    """
    return isinstance(model, RegressorMixin)
