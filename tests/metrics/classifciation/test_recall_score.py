import numpy as np

from src.metrics import recall_score


def test_recall_identical_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    expected_recall = 1.0
    recall = recall_score(y_true, y_pred)

    assert np.isclose(expected_recall, recall, atol=1e-5, rtol=1e-5)


def test_recall_different_labels():
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])

    expected_recall = 0.0
    recall = recall_score(y_true, y_pred)

    assert np.isclose(expected_recall, recall, atol=1e-5, rtol=1e-5)


def test_recall_reversed_labels():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    expected_recall = 0.0
    recall = recall_score(y_true, y_pred)

    assert np.isclose(expected_recall, recall, atol=1e-5, rtol=1e-5)


def test_recall_all_true():
    y_true = np.array([1, 0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1, 1])

    expected_recall = 1.0
    recall = recall_score(y_true, y_pred)

    assert np.isclose(expected_recall, recall, atol=1e-5, rtol=1e-5)


def test_recall_varied_true():
    y_true = np.array([1, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1])

    expected_recall = 0.5
    recall = recall_score(y_true, y_pred)

    assert np.isclose(expected_recall, recall, atol=1e-5, rtol=1e-5)
