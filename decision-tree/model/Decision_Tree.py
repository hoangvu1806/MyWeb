import numpy as np

class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
    ):
        """
        Initialize the DecisionTree with hyperparameters.
        Args:
            max_depth (int): Maximum depth of the tree. None for unlimited depth.
            min_samples_split (int): Minimum number of samples required to split.
            criterion (str): The metric to evaluate splits ('gini' or 'entropy').
            max_features (int): Number of features to consider for the best split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None

    def gini_index(self, y):
        """
        Calculate the Gini index for a set of labels.
        Args:
            y (array-like): Target labels.
        Returns:
            float: Gini impurity.
        """
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p_cls = len(y[y == cls]) / len(y)
            gini -= p_cls**2
        return gini

    def entropy(self, y):
        """
        Calculate the entropy for a set of labels.
        Args:
            y (array-like): Target labels.
        Returns:
            float: Entropy value.
        """
        classes = np.unique(y)
        entropy = 0.0
        for cls in classes:
            p_cls = len(y[y == cls]) / len(y)
            if p_cls > 0:
                entropy -= p_cls * np.log2(p_cls)  # Avoid log(0) errors
        return entropy

    def information_gain(self, left_y, right_y, current_uncertainty):
        """
        Calculate the information gain from a split.
        Args:
            left_y (array-like): Labels in the left split.
            right_y (array-like): Labels in the right split.
            current_uncertainty (float): Current impurity/uncertainty.
        Returns:
            float: Information gain from the split.
        """
        p = float(len(left_y)) / (len(left_y) + len(right_y))  # Proportion of left split
        return (
            current_uncertainty
            - p * self.calculate_uncertainty(left_y)
            - (1 - p) * self.calculate_uncertainty(right_y)
        )

    def calculate_uncertainty(self, y):
        """
        Calculate impurity based on the specified criterion.
        Args:
            y (array-like): Target labels.
        Returns:
            float: Impurity measure (Gini or Entropy).
        """
        if self.criterion == "gini":
            return self.gini_index(y)
        elif self.criterion == "entropy":
            return self.entropy(y)

    def split(self, dataset, column, value):
        """
        Split the dataset based on a feature column and a split value.
        Args:
            dataset (array-like): Combined feature and label array.
            column (int): Feature index for splitting.
            value (int/float/str): Value to split on.
        Returns:
            tuple: Left and right subsets of the dataset.
        """
        if isinstance(value, (int, float)):  # Numerical split
            left = dataset[dataset[:, column] <= value]
            right = dataset[dataset[:, column] > value]
        else:  # Categorical split
            left = dataset[dataset[:, column] == value]
            right = dataset[dataset[:, column] != value]
        return left, right

    def best_split(self, X, y, max_features=None):
        """
        Find the best feature and value to split on.
        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            max_features (int): Number of features to consider (unused here).
        Returns:
            tuple: Best gain, best column index, and best split value.
        """
        best_gain = 0
        best_split_col = None
        best_split_value = None
        current_uncertainty = self.calculate_uncertainty(y)

        for col in range(X.shape[1]):  # Iterate over all features
            values = np.unique(X[:, col])  # Unique values for this feature
            for val in values:  # Test all potential splits
                left, right = self.split(np.c_[X, y], col, val)
                if len(left) == 0 or len(right) == 0:  # Skip invalid splits
                    continue
                left_y = left[:, -1]
                right_y = right[:, -1]
                gain = self.information_gain(left_y, right_y, current_uncertainty)
                if gain > best_gain:  # Update the best split
                    best_gain, best_split_col, best_split_value = gain, col, val
        return best_gain, best_split_col, best_split_value

    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            depth (int): Current depth of the tree.
        Returns:
            tuple/int: A node represented as (column, value, left_tree, right_tree) 
                       or a leaf node (class label).
        """
        # Stop if maximum depth or pure node is reached
        if (
            len(np.unique(y)) == 1  # All labels are the same
            or depth == self.max_depth  # Reached maximum depth
            or len(y) < self.min_samples_split  # Not enough samples to split
        ):
            return np.unique(y, return_counts=True)[0][0]  # Return the majority class

        gain, split_col, split_value = self.best_split(X, y)
        if gain == 0:  # No split improves information gain
            return np.unique(y, return_counts=True)[0][0]

        left, right = self.split(np.c_[X, y], split_col, split_value)
        left_tree = self.build_tree(left[:, :-1], left[:, -1], depth + 1)  # Recur left
        right_tree = self.build_tree(right[:, :-1], right[:, -1], depth + 1)  # Recur right
        return (split_col, split_value, left_tree, right_tree)

    def fit(self, X, y):
        """
        Fit the decision tree to the dataset.
        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
        """
        self.tree = self.build_tree(np.array(X), np.array(y))

    def predict_sample(self, sample, tree):
        """
        Predict the label for a single sample using the tree.
        Args:
            sample (array-like): A single data point.
            tree (tuple/int): The decision tree or leaf node.
        Returns:
            int: Predicted class label.
        """
        if not isinstance(tree, tuple):  # If a leaf node, return class label
            return tree
        split_col, split_value, left_tree, right_tree = tree
        if isinstance(split_value, (int, float)):  # Numerical split
            if sample[split_col] <= split_value:
                return self.predict_sample(sample, left_tree)
            else:
                return self.predict_sample(sample, right_tree)
        else:  # Categorical split
            if sample[split_col] == split_value:
                return self.predict_sample(sample, left_tree)
            else:
                return self.predict_sample(sample, right_tree)

    def predict(self, X):
        """
        Predict the labels for a dataset.
        Args:
            X (array-like): Feature matrix.
        Returns:
            array: Predicted labels.
        """
        return np.array(
            [self.predict_sample(sample, self.tree) for sample in np.array(X)]
        )
