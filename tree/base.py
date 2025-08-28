%%writefile tree/base.py
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd
from tree.utils import opt_split_attribute, split_data, check_ifreal, one_hot_encoding

@dataclass
class _Node:
    feature: Optional[str] = None
    threshold: Optional[float] = None  # numeric threshold
    is_cat: bool = False
    cat_value: Optional[object] = None
    left: Optional["__class__"] = None
    right: Optional["__class__"] = None
    prediction: Optional[object] = None

class DecisionTree:
    def __init__(self, criterion: Literal["information_gain","gini_index"]="information_gain", max_depth: int = 5, min_samples_split: int = 2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self._is_reg = None
        self._categorical_cols = None

    def _leaf_value(self, y: pd.Series):
        if self._is_reg:
            return float(y.mean())
        # classification
        return y.mode(dropna=False).iloc[0]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # remember categorical columns for prediction-time consistency
        self._is_reg = check_ifreal(y)
        # Keep original types (we will split directly on columns; no one-hot here)
        self._categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.root = self._build(X.reset_index(drop=True), y.reset_index(drop=True), depth=0)

    def _build(self, X: pd.DataFrame, y: pd.Series, depth: int) -> _Node:
        node = _Node()
        # stopping conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split or y.nunique(dropna=False) == 1 or X.shape[1] == 0:
            node.prediction = self._leaf_value(y)
            return node

        best = opt_split_attribute(X, y, self.criterion, X.columns)
        if best["feat"] is None or best["gain"] <= 0:
            node.prediction = self._leaf_value(y)
            return node

        (XL, yL), (XR, yR) = split_data(X, y, best["feat"], best["cat_value"] if best["is_cat"] else best["thr"], best["is_cat"])
        if len(XL) == 0 or len(XR) == 0:
            node.prediction = self._leaf_value(y)
            return node

        node.feature = best["feat"]
        node.is_cat = best["is_cat"]
        node.threshold = None if best["is_cat"] else float(best["thr"])
        node.cat_value = best["cat_value"]
        node.left = self._build(XL, yL, depth+1)
        node.right = self._build(XR, yR, depth+1)
        return node

    def _predict_row(self, row: pd.Series, node: _Node):
        while node.prediction is None:
            val = row[node.feature]
            if node.is_cat:
                go_left = (val == node.cat_value)
            else:
                go_left = (val <= node.threshold)
            node = node.left if go_left else node.right
        return node.prediction

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = [self._predict_row(X.iloc[i], self.root) for i in range(len(X))]
        return pd.Series(preds)

    def plot(self) -> None:
        def _print(node, indent=""):
            if node.prediction is not None:
                print(indent + f"Leaf -> {node.prediction}")
                return
            cond = f"{node.feature} == {node.cat_value}" if node.is_cat else f"{node.feature} <= {node.threshold:.4f}"
            print(indent + f"?({cond})")
            print(indent + "  Y:", end=" "); _print(node.left, indent + "    ")
            print(indent + "  N:", end=" "); _print(node.right, indent + "    ")
        _print(self.root)

