%%writefile tree/utils.py
import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    # one-hot encode object/category columns; leave numeric as is
    return pd.get_dummies(X, drop_first=False)

def check_ifreal(y: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(y)

def _class_probs(Y: pd.Series):
    counts = Y.value_counts(dropna=False)
    probs = counts / counts.sum()
    return probs.values

def entropy(Y: pd.Series) -> float:
    p = _class_probs(Y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def gini_index(Y: pd.Series) -> float:
    p = _class_probs(Y)
    return float(1.0 - (p**2).sum()) if p.size else 0.0

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _variance(Y: pd.Series) -> float:
    if Y.size == 0: return 0.0
    return float(np.var(Y.values, ddof=0))

def _best_numeric_threshold(x: pd.Series, y: pd.Series, criterion: str, is_reg: bool):
    # try midpoints between sorted unique values of x
    order = np.argsort(x.values)
    x_sorted = x.values[order]
    y_sorted = y.values[order]
    uniq = np.unique(x_sorted)
    if uniq.size <= 1:
        return None, None  # no split
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0

    best_gain, best_thr = -np.inf, None
    for thr in thresholds:
        left_mask = x_sorted <= thr
        yL, yR = y_sorted[left_mask], y_sorted[~left_mask]
        if is_reg:
            base = _variance(pd.Series(y_sorted))
            child = (yL.size/ y_sorted.size)*_variance(pd.Series(yL)) + (yR.size/ y_sorted.size)*_variance(pd.Series(yR))
            gain = base - child
        else:
            if criterion == "gini_index":
                base = gini_index(pd.Series(y_sorted))
                child = (yL.size/ y_sorted.size)*gini_index(pd.Series(yL)) + (yR.size/ y_sorted.size)*gini_index(pd.Series(yR))
                gain = base - child
            else:
                base = entropy(pd.Series(y_sorted))
                child = (yL.size/ y_sorted.size)*entropy(pd.Series(yL)) + (yR.size/ y_sorted.size)*entropy(pd.Series(yR))
                gain = base - child
        if gain > best_gain:
            best_gain, best_thr = float(gain), float(thr)
    return best_gain, best_thr

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    is_reg = check_ifreal(Y)
    if is_reg:
        base = _variance(Y)
        if _is_numeric(attr):
            gain, thr = _best_numeric_threshold(attr, Y, criterion, True)
            return -np.inf if thr is None else gain
        else:
            # categorical: split by category
            total = len(Y)
            child_var = 0.0
            for cat, idx in attr.groupby(attr).groups.items():
                child_var += (len(idx)/total) * _variance(Y.iloc[list(idx)])
            return base - child_var
    else:
        if _is_numeric(attr):
            gain, thr = _best_numeric_threshold(attr, Y, criterion, False)
            return -np.inf if thr is None else gain
        else:
            total = len(Y)
            if criterion == "gini_index":
                base = gini_index(Y)
                child_imp = sum((len(idx)/total)*gini_index(Y.iloc[list(idx)])
                                for _, idx in attr.groupby(attr).groups.items())
            else:
                base = entropy(Y)
                child_imp = sum((len(idx)/total)*entropy(Y.iloc[list(idx)])
                                for _, idx in attr.groupby(attr).groups.items())
            return base - child_imp

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, features: pd.Index):
    is_reg = check_ifreal(y)
    best = {"feat": None, "thr": None, "gain": -np.inf, "is_cat": False, "cat_value": None}
    for feat in features:
        col = X[feat]
        if _is_numeric(col):
            gain, thr = _best_numeric_threshold(col, y, criterion, is_reg)
            if thr is not None and gain > best["gain"]:
                best.update({"feat": feat, "thr": thr, "gain": float(gain), "is_cat": False, "cat_value": None})
        else:
            # categorical: try one-vs-rest split for each category, pick best
            cats = col.astype("category").cat.categories
            for c in cats:
                mask = (col == c)
                yL, yR = y[mask], y[~mask]
                if is_reg:
                    base = _variance(y)
                    child = (len(yL)/len(y))*_variance(yL) + (len(yR)/len(y))*_variance(yR)
                    gain = base - child
                else:
                    if criterion == "gini_index":
                        base = gini_index(y)
                        child = (len(yL)/len(y))*gini_index(yL) + (len(yR)/len(y))*gini_index(yR)
                        gain = base - child
                    else:
                        base = entropy(y)
                        child = (len(yL)/len(y))*entropy(yL) + (len(yR)/len(y))*entropy(yR)
                        gain = base - child
                if gain > best["gain"]:
                    best.update({"feat": feat, "thr": None, "gain": float(gain), "is_cat": True, "cat_value": c})
    return best

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value, is_cat: bool):
    if is_cat:
        left_mask = (X[attribute] == value)
    else:
        left_mask = (X[attribute] <= value)
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]
    return (X_left, y_left), (X_right, y_right)
