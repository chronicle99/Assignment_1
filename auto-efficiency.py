"""
Experiments for Question 3 of the assignment: automotive efficiency.

This script uses the UCI Auto MPG dataset to train and evaluate a
custom decision tree for regression. The target variable is miles per
gallon (``mpg``) and the input features are various characteristics of
each automobile. After cleaning the data, the script splits it into
train and test sets, trains our decision tree, evaluates its
performance using RMSE and MAE, and compares the results against
scikit‑learn's ``DecisionTreeRegressor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from tree.base import DecisionTree
from metrics import rmse, mae


def load_auto_mpg() -> pd.DataFrame:
    """Load and clean the Auto MPG dataset from the UCI repository.

    Returns:
        A cleaned pandas DataFrame containing numeric features and the
        target ``mpg``. Rows with missing or invalid values are
        dropped.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    # Column names as described in the UCI documentation
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    df = pd.read_csv(url, delim_whitespace=True, names=columns, na_values="?")
    # Drop rows with missing horsepower or mpg
    df = df.dropna(subset=["horsepower", "mpg"])
    # Convert numeric columns to appropriate types
    numeric_cols = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    # Convert origin to categorical (1=USA, 2=Europe, 3=Asia)
    df["origin"] = df["origin"].astype("category")
    # Drop the car name since it's not useful for this task
    df = df.drop(columns=["car_name"])
    return df.reset_index(drop=True)


def main(random_state: int = 42) -> None:
    df = load_auto_mpg()
    # Features and target
    X = df.drop(columns=["mpg"])
    y = df["mpg"]

    # Split into 70/30 train/test sets
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # Train custom decision tree for regression
    tree = DecisionTree(criterion="information_gain", max_depth=5)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    custom_rmse = rmse(y_pred, y_test)
    custom_mae = mae(y_pred, y_test)
    print(f"Custom DecisionTree RMSE: {custom_rmse:.4f}")
    print(f"Custom DecisionTree MAE:  {custom_mae:.4f}")

    # Train scikit‑learn's DecisionTreeRegressor for comparison
    sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=random_state)
    # Need to encode categorical origin using one‑hot encoding
    X_train_enc = pd.get_dummies(X_train, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, drop_first=False)
    # Align columns of train and test
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    sklearn_tree.fit(X_train_enc, y_train)
    y_sklearn_pred = pd.Series(sklearn_tree.predict(X_test_enc))
    sklearn_rmse = rmse(y_sklearn_pred, y_test)
    sklearn_mae = mae(y_sklearn_pred, y_test)
    print(f"Sklearn DecisionTreeRegressor RMSE: {sklearn_rmse:.4f}")
    print(f"Sklearn DecisionTreeRegressor MAE:  {sklearn_mae:.4f}")

    # Plot predicted vs actual for both models
    plt.figure()
    plt.scatter(y_test, y_pred, label="Custom DT", alpha=0.7)
    plt.scatter(y_test, y_sklearn_pred, label="Sklearn DT", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Auto MPG Regression: Actual vs Predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()