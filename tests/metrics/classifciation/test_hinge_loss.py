import numpy as np
from src.metrics import hinge_loss

def test_hingeloss_correct_scores():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([100, -100, 100, -100, 100])

    expect_hinge_loss = 0.0
    hingeloss = hinge_loss(y_true, y_pred)
    assert np.isclose(expect_hinge_loss, hingeloss, atol=1e-5, rtol=1e-5)


def test_hingeloss_reverse_scores():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([-100, 100, -100, 100, -100])

    expect_hinge_loss = 101.0
    hingeloss = hinge_loss(y_true, y_pred)
    assert np.isclose(expect_hinge_loss, hingeloss, atol=1e-5, rtol=1e-5)


def test_hingeloss_zero():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])

    expect_hinge_loss = 1.0
    hingeloss = hinge_loss(y_true, y_pred)
    assert np.isclose(expect_hinge_loss, hingeloss, atol=1e-5, rtol=1e-5)
