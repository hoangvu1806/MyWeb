import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# Các hàm chuẩn hóa dữ liệu nha Tú ml
def min_max_scaling(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    scale = max_val - min_val
    scale[scale == 0] = 1  # Tránh chia cho 0
    return (X - min_val) / scale

def standard_scaling(X):
    mean_val = np.mean(X, axis=0)
    std_val = np.std(X, axis=0)
    std_val[std_val == 0] = 1
    return (X - mean_val) / std_val

def robust_scaling(X):
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    return (X - median) / iqr

def remove_outliers(df, threshold=1.5):
    # Tạo bản sao của DataFrame để không làm thay đổi dữ liệu gốc
    df_no_outliers = df.copy()
    # Xác định cột số trong DataFrame
    numeric_columns = df_no_outliers.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        # Tính toán Q1 (25th percentile) và Q3 (75th percentile)
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        # Tính IQR (Interquartile Range)
        IQR = Q3 - Q1
        # Xác định các giới hạn để loại bỏ outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # Loại bỏ các hàng có giá trị ngoài phạm vi [lower_bound, upper_bound]
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
        ]
    return df_no_outliers

# Hàm tiền xử lý dữ liệu
def preprocess_data(
    file_path,
    target_column="target",
    split_ratio=0.8,
    seed=42,
    drop_columns=[],
    scaling_method="min_max",
):
    if seed is not None:
        np.random.seed(seed)
    # Load dataset nha Tú ml
    df = pd.read_csv(file_path, sep=";", engine="python").dropna()
    # In ra tên cột để kiểm tra nha Tú ml
    print("Columns in the DataFrame:", df.columns.tolist())
    # Drop các cột không cần thiết (nếu có)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    # Kiểm tra cột mục tiêu có tồn tại hay không
    df = remove_outliers(df)
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame columns: {df.columns.tolist()}"
        )
    # Lưu giá trị cột mục tiêu trước khi xử lý
    y = df[target_column].values
    # Xóa cột mục tiêu khỏi DataFrame
    df = df.drop(columns=[target_column])
    # Xử lý các cột không phải số với one-hot encoding nha Tú ml
    df = pd.get_dummies(df)
    # Đảm bảo tất cả dữ liệu là dạng float sau khi mã hóa
    df = df.astype(float)
    X = df.values
    # Chọn phương pháp chuẩn hóa nha Tú ml
    if scaling_method == "min_max":
        X = min_max_scaling(X)
    elif scaling_method == "standard":
        X = standard_scaling(X)
    elif scaling_method == "robust":
        X = robust_scaling(X)
    # Shuffle the dataset
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    split_idx = int(split_ratio * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


# Linear Regression class
class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.loss_history = []

    def cost_function(self, prediction, ground_truth):
        return np.mean((prediction - ground_truth) ** 2)

    # Compute the cost (Mean Squared Error)
    def loss_function(self, X, y):
        m = len(y)
        predictions = X.dot(self.weights)
        error = self.cost_function(predictions, y)
        loss = (1 / (2 * m)) * np.sum(error)
        return loss

    # Gradient descent algorithm to optimize the parameters
    def gradient_descent(self, X, y):
        m = len(y)
        for _ in tqdm(range(self.num_iterations), desc="Iterations", leave=False):
            predictions = X.dot(self.weights)
            error = predictions - y
            gradient = (1 / m) * X.T.dot(error)
            self.weights -= self.learning_rate * gradient
        return self.weights

    # Train the model
    def fit(self, X, y, epochs=1):
        self.weights = np.zeros(X.shape[1])
        for _ in tqdm(range(epochs), desc="Epochs"):
            self.loss_history.append(self.loss_function(X, y))
            self.gradient_descent(X, y)

    # Predict new data
    def predict(self, X):
        if self.weights is None:
            raise ValueError(
                "Model has not been trained yet. Please call 'fit' to train the model first."
            )
        return X.dot(self.weights)


if __name__ == "__main__":
    file_path = "D:\MyWeb\linear-regression\datasets\winequality.csv"  # Replace with your actual file path
    target_column = 'quality'  # Chọn cột target
    drop_columns = []  # Bỏ cột không cần thiết
    learning_rate = 0.1
    epochs = 1000
    X_train, X_test, y_train, y_test = preprocess_data(
        file_path,
        target_column,
        split_ratio=0.8,
        seed=42,
        drop_columns=drop_columns,
        scaling_method="min_max",
    )
    # Initialize and train the Linear Regression model
    model = LinearRegression(learning_rate=learning_rate, num_iterations=1)

    # Train the model with 10 epochs
    model.fit(X_train, y_train, epochs)
    print("Learning rate: ", learning_rate)
    print("Epochs: ", epochs)
    print("Lastest Loss: ", model.loss_history[-1])
    print("Weights:", model.weights)
    # Predict on test data
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("mse: ", mse)
    # Plot the loss history
    # -----Uisng libraries-----
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score, r2_score

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("mse: ", mse)
