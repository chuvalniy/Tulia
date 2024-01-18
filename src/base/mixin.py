from abc import ABC


class Mixin(ABC):
    """
    Interface to define all mixins.
    """
    pass


class ClassifierMixin(Mixin):
    """
    Mixin to define all classification models.
    """
    pass


class RegressorMixin(Mixin):
    """
    Mixin to define all regression models.
    """
    pass
