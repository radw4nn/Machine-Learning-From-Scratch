import numpy as numpy
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def check_leaf_node(self):
        return self.value is not None


class DecisionTreeModel:
    def __init__(self, min_split_samples=2, max_tree_depth=100, num_features=None):
        self.min_split_samples = min_split_samples
        self.max_tree_depth = max_tree_depth
        self.num_features = num_features
        self.root = None

    def train_model(self, X_data, y_labels):
        self.num_features = X_data.shape[1] if not self.num_features else min(X_data.shape[1], self.num_features)
        self.root = self._grow_decision_tree(X_data, y_labels)

    def _grow_decision_tree(self, X_data, y_labels, depth=0):
        num_samples, num_feats = X_data.shape
        num_labels = len(numpy.unique(y_labels))

        if depth >= self.max_tree_depth or num_labels == 1 or num_samples < self.min_split_samples:
            leaf_value = self._most_frequent_label(y_labels)
            return TreeNode(value=leaf_value)

        feat_indices = numpy.random.choice(num_feats, self.num_features, replace=False)

        best_feature, best_threshold = self._find_best_split(X_data, y_labels, feat_indices)

        left_indices, right_indices = self._split_data(X_data[:, best_feature], best_threshold)
        left = self._grow_decision_tree(X_data[left_indices, :], y_labels[left_indices], depth + 1)
        right = self._grow_decision_tree(X_data[right_indices, :], y_labels[right_indices], depth + 1)
        return TreeNode(best_feature, best_threshold, left, right)

    def _find_best_split(self, X_data, y_labels, feat_indices):
        best_gain = -1
        split_index, split_threshold = None, None

        for feat_index in feat_indices:
            X_column = X_data[:, feat_index]
            thresholds = numpy.unique(X_column)

            for threshold in thresholds:
                gain = self._calculate_information_gain(y_labels, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _calculate_information_gain(self, y_labels, X_column, threshold):
        parent_entropy = self._calculate_entropy(y_labels)

        left_indices, right_indices = self._split_data(X_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y_labels)
        n_left, n_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._calculate_entropy(y_labels[left_indices]), self._calculate_entropy(y_labels[right_indices])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split_data(self, X_column, split_threshold):
        left_indices = numpy.argwhere(X_column <= split_threshold).flatten()
        right_indices = numpy.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _calculate_entropy(self, y_labels):
        hist = numpy.bincount(y_labels)
        probabilities = hist / len(y_labels)
        return -numpy.sum([p * numpy.log(p) for p in probabilities if p > 0])

    def _most_frequent_label(self, y_labels):
        counter = Counter(y_labels)
        value = counter.most_common(1)[0][0]
        return value

    def make_predictions(self, X_data):
        return numpy.array([self._traverse_decision_tree(x, self.root) for x in X_data])

    def _traverse_decision_tree(self, x, node):
        if node.check_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_decision_tree(x, node.left)
        return self._traverse_decision_tree(x, node.right)
