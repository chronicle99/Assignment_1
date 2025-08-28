%%writefile metrics.py
from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    # y_hat and y must have same length
    return float((y_hat.reset_index(drop=True) == y.reset_index(drop=True)).mean())


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
   

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    diff = y_hat.reset_index(drop=True) - y.reset_index(drop=True)
    # no numpy import: square via **2, then mean, then square-root via **0.5
    return float(((diff ** 2).mean()) ** 0.5)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    diff = (y_hat.reset_index(drop=True) - y.reset_index(drop=True)).abs()
    return float(diff.mean())
    
