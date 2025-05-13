import sys, os
sys.path.append(os.path.abspath("./decision-tree"))
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.Decision_Tree import DecisionTree

def tree_to_json(tree, column_names, depth=0, max_depth=None):
    """
    Convert a decision tree to a JSON-serializable dictionary with column names.

    Args:
        tree (tuple): The decision tree structure to convert.
        column_names (list): List of column names corresponding to feature indices.
        depth (int): The current depth of the tree (default: 0).
        max_depth (int): The maximum depth to traverse the tree (default: None for no limit).

    Returns:
        dict: JSON-serializable dictionary representation of the tree.
    """
    # Handle leaf nodes or max depth
    if not isinstance(tree, tuple) or (max_depth is not None and depth >= max_depth):
        # Ensure the label is JSON-serializable
        return {"label": convert_to_serializable(tree)}

    # Unpack the tree structure
    split_feature, split_value, left_branch, right_branch = tree

    # Ensure split_value is JSON-serializable
    split_value = convert_to_serializable(split_value)

    # Get the feature name using column_names
    feature_name = column_names[split_feature]

    # Recursively process left and right branches
    left_json = tree_to_json(left_branch, column_names, depth + 1, max_depth)
    right_json = tree_to_json(right_branch, column_names, depth + 1, max_depth)

    # Return the current node as a JSON-serializable dictionary
    return {
        "feature": feature_name,
        "value": split_value,
        "left": left_json,
        "right": right_json,
    }


def convert_to_serializable(value):
    """
    Convert a value to a JSON-serializable type.

    Args:
        value: The value to convert.

    Returns:
        The value converted to a JSON-serializable type.
    """
    if isinstance(value, (np.generic, np.ndarray)):
        return value.item()  # Convert NumPy scalar or single-element array to Python scalar
    elif isinstance(value, list):
        return [convert_to_serializable(v) for v in value]  # Process each element in a list
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}  # Process dictionary values
    elif isinstance(value, tuple):
        return tuple(convert_to_serializable(v) for v in value)  # Process tuple values
    return value  # Return original value if already JSON-serializable


def remove_outliers(df, threshold=1.5):
    df_no_outliers = df.copy()
    numeric_columns = df_no_outliers.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
        ]
    return df_no_outliers

def preprocess_data(
    file_path, target_column="target", split_ratio=0.8, seed=42, drop_columns=[]
):

    if seed is not None:
        np.random.seed(seed)
    # Load dataset
    df = pd.read_csv(file_path, sep=",|;", engine="python")
    df = df.dropna()
    if drop_columns:
        df = df.drop(columns=drop_columns)
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    print("Columns in the DataFrame:", df.columns.tolist())

    indices = np.random.permutation(len(X))
    split_index = int(X.shape[0] * split_ratio)

    X_train = X.iloc[indices[split_index:]]
    X_test = X.iloc[indices[:split_index]]
    y_train = y.iloc[indices[split_index:]]
    y_test = y.iloc[indices[:split_index]]

    return X_train, X_test, y_train, y_test

def main(file, target_column, drop_columns=[], max_depth=10, min_samples_split=2, criterion="gini"):
    print("Decision function: Compute the decision function!")
    SEED = 42
    split_ratio = 0.7
    X_train, X_test, y_train, y_test = preprocess_data(
        file, target_column, 0.7, 42, drop_columns
    )
    model = DecisionTree(max_depth, min_samples_split, criterion)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    print(f"Criterion: {criterion}")
    print(f"Max depth: {max_depth}")
    print(f"Min depth: {min_samples_split}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    tree = tree_to_json(model.tree, X_train.columns, max_depth=max_depth)
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    return tree, list(X_train.columns), metrics, split_ratio, SEED

