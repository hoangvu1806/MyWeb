import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.Linear_Regression import LinearRegression as LR
from model.Decision_Tree import DecisionTree as DT


def min_max_scaling(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)


def standard_scaling(X):
    mean_val = np.mean(X, axis=0)
    std_val = np.std(X, axis=0)
    std_val[std_val == 0] = 1
    return (X - mean_val) / std_val


def robust_scaling(X):
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    return (X - median) / iqr


# Preprocess function: normalizes the data, handles non-numeric features, and adds bias term
def preprocess_data(
    file_path, target_column="charges", split_ratio=0.8, seed=None, drop_columns=None
):
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    # Load the dataset
    df = pd.read_csv(file_path, delimiter=r"[;,]", engine="python")
    # Drop unwanted columns if provided

    # Handle non-numeric columns using one-hot encoding
    df = pd.get_dummies(df)

    # Convert boolean values to integers (0 and 1)
    df = df.astype(float)

    # Define features (X) and target (y)
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # Add a column of ones to X for the bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # Chuẩn hóa dữ liệu
    X[:, 1:] = min_max_scaling(X[:, 1:])
    # print(X.T)
    # Shuffle the dataset
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Split the dataset into training and testing sets
    split_idx = int(split_ratio * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "./datasets/insurance.csv"  # Replace with your actual file path
    target_column = "charges"  # Chọn target
    drop_columns = ["region", "sex", "children"]  # Bỏ cột không cần thiết
    learning_rate = 0.01
    epochs = 5000
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path, target_column, split_ratio=0.7, seed=423, drop_columns=drop_columns
    )
    model = LR()
