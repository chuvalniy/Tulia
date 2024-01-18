import numpy as np

from src.metrics import log_loss


def test_logloss_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expect_log_loss = 0.0
    logloss = log_loss(y_true, y_pred)
    assert np.isclose(expect_log_loss, logloss, atol=1e-5, rtol=1e-5)


def test_logloss_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expect_log_loss = -np.log(1e-8)
    logloss = log_loss(y_true, y_pred)
    assert np.isclose(expect_log_loss, logloss, atol=1e-5, rtol=1e-5)


def test_logloss_equal_prob():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    expect_log_loss = -np.log(0.5)
    logloss = log_loss(y_true, y_pred)
    assert np.isclose(expect_log_loss, logloss, atol=1e-5, rtol=1e-5)


def test_logloss_close_to_target():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.1, 0.1, 0.9])

    expect_log_loss = -np.log(0.9)
    logloss = log_loss(y_true, y_pred)
    assert np.isclose(expect_log_loss, logloss, atol=1e-5, rtol=1e-5)


def test_logloss_different_predictions():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([0.83, 0.11, 0.02, 0.17, 0.97])

    expect_log_loss = 0.53985483159 / 5
    logloss = log_loss(y_true, y_pred)
    assert np.isclose(expect_log_loss, logloss, atol=1e-5, rtol=1e-5)
