from typing import Union

import numpy as np

from src.base import Model


class _XGBTreeNode:
    def __init__(
            self,
            left: '_XGBTreeNode' = None,
            right: '_XGBTreeNode' = None,
            feature: int = None,
            threshold: float = None,
            prediction: int = None,
            gain: float = None
    ):
        """
        :param left: Left node.
        :param right: Right node.
        :param feature: Feature by which data is divided.
        :param threshold: Threshold to split the data.
        :param prediction: Prediction to a class.
        :param gain: Gain for the current node.
        """

        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.prediction = prediction
        self.gain = gain

    def is_leaf_node(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return f"_XGBTreeNode(" \
               f"left={self.left}," \
               f" right={self.right}," \
               f" feature={self.feature}," \
               f" threshold={self.threshold}," \
               f" prediction={self.prediction}," \
               f" gain={self.gain}" \
               f")"


# TODO: Replace this tree with abstract of Decision and XGB trees.
class XGBRegressionTree(Model):
    """
    Regression tree for XGB algorithm.
    """

    def __init__(
            self,
            max_depth: int = 3,
            min_samples_split: int = 2,
            max_features: float = 1.0,
            reg_lambda: float = 1.0,
            gamma: float = 0.0
    ):
        """
        :param max_depth: Maximum depth of a decision tree.
        :param min_samples_split: Minimum number of samples to split data into right and left nodes.
        :param max_features: Percentage of features to use for training.
        :param reg_lambda: Regularization parameter for tree split.
        :param gamma: Regularization parameter to prune a tree.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self.root = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train XGBoost Decision Tree.
        :param x: Training data.
        :param y: Targets.
        :return:
        """
        # Create new root node.
        self.root = _XGBTreeNode()

        feature_idxs = self._select_features(x)

        # Create queue to traverse through a decision tree using breadth-first search.
        queue = [(self.root, x, y)]
        for depth in range(self.max_depth):
            queue_len = len(queue)
            for _ in range(queue_len):
                curr_node, data, targets = queue.pop(0)
                n_samples, n_classes = len(targets), len(np.unique(targets))

                feature_idx, threshold, gain = self._find_split(data, targets, feature_idxs)

                # Update the current node and create left and right nodes.
                curr_node.feature = feature_idx
                curr_node.threshold = threshold
                curr_node.gain = gain

                curr_node.prediction = self._predict(targets)

                # Stopping criteria.
                if (depth == self.max_depth - 1) or (n_classes == 1) or (n_samples < self.min_samples_split):
                    continue

                curr_node.left = _XGBTreeNode()
                curr_node.right = _XGBTreeNode()

                # Initialize mask to split the data into two parts.
                mask = data[:, feature_idx] > threshold

                # Split data into left and right parts and add them to the queue.
                right_data, right_targets = data[~mask], targets[~mask]
                queue.append((curr_node.right, right_data, right_targets))

                left_data, left_targets = data[mask], targets[mask]
                queue.append((curr_node.left, left_data, left_targets))

        self._prune(self.root)

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly select features to train the model.
        :param x: Training data.
        :return: Selected features (as indices).
        """
        n_samples, n_features = x.shape

        n_selected_features = int(np.round(self.max_features * n_features))

        feature_idxs = np.random.choice(a=n_features, size=n_selected_features, replace=False)
        return feature_idxs

    def _find_split(self, x: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> (int, float, float):
        """
        Find the best feature and threshold to split data into two parts.
        :param x: Training data.
        :param y: Targets.
        :param feature_idxs: Features to use in training.
        :return: (Index of the best feature, Threshold of the best feature)
        """
        best_feature_idx, best_threshold = None, None

        best_gain = float('-inf')
        for idx in feature_idxs:
            x_col = x[:, idx]

            thresholds = np.unique(x_col)
            for threshold in thresholds:
                gain = self._calculate_gain(x_col, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def _calculate_gain(self, x: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Measure the quality of a split.
        :param x: One-dimensional array of samples.
        :param y: Targets.
        :param threshold: Threshold to split 'x' into left and right sub-arrays.
        :return: Criterion.
        """
        mask = x > threshold

        root_sim_score = np.sum(y) ** 2 / (len(y) + self.reg_lambda)

        left_data = y[mask]
        left_sim_score = np.sum(left_data) ** 2 / (len(left_data) + self.reg_lambda)

        right_data = y[~mask]
        right_sim_score = np.sum(right_data) ** 2 / (len(right_data) + self.reg_lambda)

        gain = left_sim_score + right_sim_score - root_sim_score
        return gain

    def _prune(self, root: _XGBTreeNode) -> bool:
        """
        Remove tree branches depending on the gain and gamma regularization parameter.
        :param root: XGBoost root node.
        """
        if root is None:
            return False

        left = self._prune(root.left)
        right = self._prune(root.right)

        # Remove branch.
        if left and right:
            root.left = None
            root.right = None

        if root.gain < self.gamma and root.left is None and root.right is None:
            return True

        return False

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict classes by traversing over a tree for every data sample.
        :param x: Test data.
        :return: Test predictions.
        """
        predictions = np.array([self._dfs(x_sample, self.root) for x_sample in x])
        return predictions

    def _dfs(self, x: np.ndarray, root: _XGBTreeNode):
        """
        Depth-first search traversal over a tree.
        :param x: Sample from the data.
        :param root: Tree node.
        :return: Prediction for a sample.
        """
        if root.is_leaf_node():
            return root.prediction

        if x[root.feature] > root.threshold:
            return self._dfs(x, root.left)

        return self._dfs(x, root.right)

    def _predict(self, y: np.ndarray) -> Union[np.ndarray, float, int]:
        """
        Calculate similarity score.
        :param y: Targets.
        :return: Most common class.
        """
        prediction = np.sum(y) / (len(y) + self.reg_lambda)
        return prediction


class XGBRegressor(Model):
    """
    XGBoost for regression tasks.
    """

    def __init__(
            self,
            learning_rate: float = 3e-1,
            n_steps: int = 100,
            max_depth: int = 3,
            min_samples_split: int = 2,
            max_features: float = 1.0,
            reg_lambda: float = 1.0,
            gamma: float = 0.0
    ):
        """
        :param learning_rate: Learning rate for gradient descent.
        :param n_steps: Number of gradient descent steps.
        :param max_depth: Maximum depth of a decision tree.
        :param min_samples_split: Minimum number of samples to split data into right and left nodes.
        :param max_features: Percentage of features to use for training.
        :param reg_lambda: Regularization parameter for tree split.
        :param gamma: Regularization parameter to prune a tree.
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_steps
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self.constant_prediction = None
        self.trees = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train gradient boosting.
        :param x: Training data.
        :param y: Target feature.
        :return:
        """

        # Initial prediction
        self.constant_prediction = np.mean(y)

        prediction = self.constant_prediction
        for _ in range(self.n_estimators):
            residuals = y - prediction

            tree = XGBRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )

            tree.fit(x, residuals)
            prediction = prediction + self.learning_rate * tree.predict(x)

            self.trees.append(tree)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target feature using pretrained boosting trees.
        :param x: Test data.
        :return: Test predictions.
        """
        predictions = self._predict(x)
        return predictions

    def _predict(self, x: np.ndarray) -> Union[np.ndarray, float, int]:
        """
        Calculate targets using tree predictions.
        :param x: Input array.
        :return: Predictions.
        """
        n_samples, _ = x.shape

        predictions = np.ones(n_samples) * self.constant_prediction
        for tree in self.trees:
            predictions = predictions + self.learning_rate * tree.predict(x)

        return predictions
