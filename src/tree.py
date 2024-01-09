import numpy as np

from model import Model


class _DecisionTreeNode:
    def __init__(
            self,
            left: '_DecisionTreeNode' = None,
            right: '_DecisionTreeNode' = None,
            feature: int = None,
            threshold: float = None,
            prediction: int = None
    ):
        """
        :param left: Left node.
        :param right: Right node.
        :param feature: Feature by which data is divided.
        :param threshold: Threshold to split the data.
        :param prediction: Prediction to a class.
        """

        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.prediction = prediction

    def is_lead_node(self):
        return self.prediction is not None


class DecisionTreeClassifier(Model):
    """
    Decision Tree for a binary classification.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, max_features: float = 1.0):
        """
        :param max_depth: Maximum depth of a decision tree.
        :param min_samples_split: Minimum number of samples to split data into right and left nodes.
        :param max_features: Percentage of features to use for training.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.root = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train a decision tree using an entropy criterion.
        :param x: Training data.
        :param y: Targets.
        :return:
        """
        # Create new root node.
        self.root = _DecisionTreeNode()

        feature_idxs = self._select_features(x)

        # Create queue to traverse through a decision tree using breadth-first search.
        queue = [(self.root, x, y)]
        for depth in range(self.max_depth):
            queue_length = len(queue)
            for _ in range(queue_length):
                curr_node, data, targets = queue.pop(0)
                n_samples, n_classes = targets.shape[0], len(np.unique(targets))

                feature_idx, threshold = self._find_split(data, targets, feature_idxs)
                # Update the current node and create left and right nodes.
                curr_node.threshold = threshold
                curr_node.feature = feature_idx

                # Stopping criteria.
                if (depth == self.max_depth - 1) or (n_classes == 1) or (n_samples < self.min_samples_split):
                    curr_node.prediction = self._find_most_common(targets)
                    continue

                curr_node.left = _DecisionTreeNode()
                curr_node.right = _DecisionTreeNode()

                # Initialize mask to split the data into two parts.
                mask = data[:, feature_idx] > threshold

                # Split data into left and right parts and add them to the queue.
                right_data, right_targets = data[~mask], targets[~mask]
                queue.append((curr_node.right, right_data, right_targets))

                left_data, left_targets = data[mask], targets[mask]
                queue.append((curr_node.left, left_data, left_targets))

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly select features to train the model.
        :param x: Training data.
        :return: Selected features (as indices).
        """
        _, n_features = x.shape
        n_features_reduced = int(np.round(n_features * self.max_features))

        feature_idxs = np.random.choice(a=n_features, size=n_features_reduced, replace=False)

        return feature_idxs

    def _find_split(self, x: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> (int, float):
        """
        Find the best data split using information gain by iterating over each column and sample in that column of the
        training data.
        :param x: Training data.
        :param y: Targets.
        :param feature_idxs: Features to use in training.
        :return: (Index of the best feature, Threshold of the best feature)
        """

        best_feature_idx, best_threshold = None, None
        best_gain = 0.0
        for feat_idx in feature_idxs:
            x_col = x[:, feat_idx]

            thresholds = np.unique(x_col)  # Get all possible values of 'x' column.
            for threshold in thresholds:
                gain = self._calculate_information_gain(x_col, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _calculate_information_gain(self, x_col: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Calculates information gain for the data split using parent and conditional entropy.
        :param x_col: One-dimensional array of samples.
        :param y: Targets.
        :param threshold: Threshold to split 'x' into left and right sub-arrays.
        :return: Information gain.
        """
        n_samples = y.shape[0]

        parent_entropy = self._calculate_entropy(y)

        # Split data into left and right parts.
        mask = x_col > threshold

        left_data = y[mask]
        left_n_samples = left_data.shape[0]

        right_data = y[~mask]
        right_n_samples = right_data.shape[0]

        # Calculate conditional entropy using left and right nodes.
        left_entropy = self._calculate_entropy(left_data)
        right_entropy = self._calculate_entropy(right_data)

        conditional_entropy = left_n_samples / n_samples * left_entropy + right_n_samples / n_samples * right_entropy

        information_gain = parent_entropy - conditional_entropy
        return information_gain

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy for a given array of targets.
        :param y: Targets.
        :return: Entropy.
        """
        n_samples = y.shape[0]

        # Count each class occurrence and calculate overall class probability.
        unique_values, class_counts = np.unique(y, return_counts=True)

        if len(unique_values) <= 1:
            return 0

        class_probs = class_counts / n_samples

        entropy = 0.0
        for prob in class_probs:
            entropy -= prob * np.log2(prob)

        return entropy

    def _find_most_common(self, y: np.ndarray) -> int:
        """
        Find the most common class in the array of targets.
        :param y: Targets.
        :return: Most common class.
        """
        most_common = np.bincount(y).argmax()
        return most_common

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict classes by traversing over the decision tree for every data sample.
        :param x: Test data.
        :return: Test predictions.
        """
        predictions = [self._dfs(x_sample, self.root) for x_sample in x]
        return np.array(predictions)

    def _dfs(self, x_sample: np.ndarray, root: _DecisionTreeNode):
        """
        Depth-first search traversal over the decision tree.
        :param x_sample: Sample from the data.
        :param root: Tree node.
        :return: Prediction for a sample.
        """
        if root.is_lead_node():
            return root.prediction

        if x_sample[root.feature] > root.threshold:
            return self._dfs(x_sample, root.left)

        return self._dfs(x_sample, root.right)
