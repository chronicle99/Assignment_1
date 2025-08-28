"""
Experiments for Question 2 of the assignment.

This script demonstrates the use of the custom decision tree on a
synthetic binary classification dataset generated via
``sklearn.datasets.make_classification``. Two experiments are
performed:

1. **Train/test split** – the dataset is split into a 70/30 train/test
   split. A decision tree is trained on the training data and its
   accuracy, per‑class precision and recall are reported on the test
   data.

2. **Nested cross‑validation** – 5‑fold cross‑validation is used to
   estimate the generalisation performance of the tree. An inner loop
   cross‑validates over a range of depths (1–8) to select the best
   depth and an outer loop evaluates the chosen model. The average
   cross‑validated accuracy for each depth is printed and the optimal
   depth is reported.

The purpose of nested cross‑validation here is to avoid optimistically
biased estimates that can arise if the depth is tuned on the same data
used for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from tree.base import DecisionTree
from metrics import accuracy, precision, recall


def train_test_split_experiment(random_state: int = 42) -> None:
    """Run a simple train/test experiment on synthetic data."""
    # Generate data with two informative features
    X_array, y_array = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=2,
        class_sep=0.5,
    )
    X = pd.DataFrame(X_array, columns=["x1", "x2"])
    y = pd.Series(y_array, dtype="category")

    # Plot the generated data
    plt.figure()
    plt.title("Synthetic classification dataset")
    plt.scatter(X["x1"], X["x2"], c=y_array, cmap="bwr", alpha=0.7)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    # Split data into 70% train, 30% test
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # Train a decision tree
    tree = DecisionTree(criterion="information_gain", max_depth=5)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    # Evaluate metrics
    acc = accuracy(y_pred, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    for cls in y.unique():
        prec = precision(y_pred, y_test, cls)
        rec = recall(y_pred, y_test, cls)
        print(f"Class {cls}: Precision = {prec:.4f}, Recall = {rec:.4f}")


def nested_cross_validation(random_state: int = 42) -> None:
    """Perform nested cross‑validation to select the optimal tree depth."""
    # Generate the same synthetic dataset
    X_array, y_array = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=2,
        class_sep=0.5,
    )
    X = pd.DataFrame(X_array, columns=["x1", "x2"])
    y = pd.Series(y_array, dtype="category")

    # Use fewer folds and depth candidates for faster execution
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    depth_candidates = list(range(1, 6))  # depths 1 through 5
    depth_performance: Dict[int, List[float]] = {d: [] for d in depth_candidates}

    for train_outer_idx, test_outer_idx in outer_cv.split(X):
        X_outer_train = X.iloc[train_outer_idx].reset_index(drop=True)
        y_outer_train = y.iloc[train_outer_idx].reset_index(drop=True)
        X_outer_test = X.iloc[test_outer_idx].reset_index(drop=True)
        y_outer_test = y.iloc[test_outer_idx].reset_index(drop=True)

        # Inner cross‑validation to choose depth
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        depth_scores: Dict[int, List[float]] = {d: [] for d in depth_candidates}
        for train_inner_idx, val_inner_idx in inner_cv.split(X_outer_train):
            X_inner_train = X_outer_train.iloc[train_inner_idx].reset_index(drop=True)
            y_inner_train = y_outer_train.iloc[train_inner_idx].reset_index(drop=True)
            X_inner_val = X_outer_train.iloc[val_inner_idx].reset_index(drop=True)
            y_inner_val = y_outer_train.iloc[val_inner_idx].reset_index(drop=True)

            for depth in depth_candidates:
                tree = DecisionTree(criterion="information_gain", max_depth=depth)
                tree.fit(X_inner_train, y_inner_train)
                y_val_pred = tree.predict(X_inner_val)
                depth_scores[depth].append(accuracy(y_val_pred, y_inner_val))

        # Determine the best depth as the one with highest average inner CV accuracy
        avg_inner_scores = {d: np.mean(depth_scores[d]) for d in depth_candidates}
        best_depth = max(avg_inner_scores, key=avg_inner_scores.get)

        # Train using the best depth on the outer training set and evaluate on the outer test set
        final_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
        final_tree.fit(X_outer_train, y_outer_train)
        y_outer_pred = final_tree.predict(X_outer_test)
        depth_performance[best_depth].append(accuracy(y_outer_pred, y_outer_test))

    # Report average outer fold performance per depth
    print("Nested cross‑validation results:")
    for depth in depth_candidates:
        scores = depth_performance[depth]
        if scores:
            print(f"Depth {depth}: Mean accuracy = {np.mean(scores):.4f} \u00b1 {np.std(scores):.4f} over {len(scores)} outer folds")
        else:
            print(f"Depth {depth}: Not selected by any outer fold")


if __name__ == "__main__":
    # Run the experiments when executed as a script
    print("Running train/test split experiment...")
    train_test_split_experiment()
    print("\nRunning nested cross‑validation experiment...")
    nested_cross_validation()