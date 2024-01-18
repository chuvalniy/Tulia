import numpy as np

from src.metrics import accuracy_score


def test_acc_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_accuracy = 1.0
    accuracy = accuracy_score(y_true, y_pred)

    assert np.isclose(expected_accuracy, accuracy, atol=1e-5, rtol=1e-5)


def test_acc_different_labels():
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])

    expected_accuracy = 0.0
    accuracy = accuracy_score(y_true, y_pred)

    assert np.isclose(expected_accuracy, accuracy, atol=1e-5, rtol=1e-5)


def test_acc_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_accuracy = 0.0
    accuracy = accuracy_score(y_true, y_pred)

    assert np.isclose(expected_accuracy, accuracy, atol=1e-5, rtol=1e-5)


def test_acc_varied_labels():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 1])

    expected_accuracy = 0.6
    accuracy = accuracy_score(y_true, y_pred)

    assert np.isclose(expected_accuracy, accuracy, atol=1e-5, rtol=1e-5)
