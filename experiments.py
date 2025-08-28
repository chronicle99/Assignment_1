"""
Runtime complexity experiments for Question 4 of the assignment.

This module investigates how the training and prediction time of the
custom decision tree scales with the number of samples ``N`` and the
number of binary features ``M``. Four types of decision trees are
considered:

1. **Real input, real output** – regression on continuous features.
2. **Real input, discrete output** – classification on continuous features.
3. **Discrete input, real output** – regression on categorical features.
4. **Discrete input, discrete output** – classification on categorical
   features.

For each combination of ``N`` and ``M`` values, synthetic datasets are
generated and the average time to call ``fit`` and ``predict`` is
measured over multiple runs. Results are plotted to illustrate how
runtime grows as the dataset size increases.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Tuple

from tree.base import DecisionTree


def generate_data(N: int, M: int, input_type: str, output_type: str, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset.

    Args:
        N: Number of samples.
        M: Number of features.
        input_type: Either ``"real"`` or ``"discrete"`` to choose feature
            types.
        output_type: Either ``"real"`` or ``"discrete"`` to choose target
            type.
        rng: A numpy random generator.

    Returns:
        A tuple ``(X, y)`` where ``X`` is an ``N × M`` DataFrame and
        ``y`` is a Series of length ``N``.
    """
    if input_type == "real":
        X = pd.DataFrame(rng.random((N, M)))
    else:
        # Discrete features: integers between 0 and 1
        X = pd.DataFrame(rng.integers(0, 2, size=(N, M))).astype("category")
    if output_type == "real":
        y = pd.Series(rng.random(N))
    else:
        y = pd.Series(rng.integers(0, 2, size=N)).astype("category")
    return X, y


def measure_time(
    N_values: List[int],
    M_values: List[int],
    input_type: str,
    output_type: str,
    num_runs: int = 5,
) -> Dict[str, np.ndarray]:
    """Measure average fit and predict times for different data sizes.

    Args:
        N_values: List of sample sizes to test.
        M_values: List of feature counts to test.
        input_type: ``"real"`` or ``"discrete"``.
        output_type: ``"real"`` or ``"discrete"``.
        num_runs: Number of repeated runs to average timing.

    Returns:
        A dictionary with keys ``"fit"`` and ``"predict"`` mapping to
        arrays of shape ``(len(N_values), len(M_values))`` storing the
        average runtimes.
    """
    rng = np.random.default_rng(42)
    fit_times = np.zeros((len(N_values), len(M_values)))
    pred_times = np.zeros((len(N_values), len(M_values)))
    for i, N in enumerate(N_values):
        for j, M in enumerate(M_values):
            ft_list: List[float] = []
            pt_list: List[float] = []
            for _ in range(num_runs):
                X, y = generate_data(N, M, input_type, output_type, rng)
                tree = DecisionTree(criterion="information_gain", max_depth=5)
                # Measure fit time
                t0 = time.perf_counter()
                tree.fit(X, y)
                t1 = time.perf_counter()
                ft_list.append(t1 - t0)
                # Measure predict time
                t0 = time.perf_counter()
                tree.predict(X)
                t1 = time.perf_counter()
                pt_list.append(t1 - t0)
            fit_times[i, j] = np.mean(ft_list)
            pred_times[i, j] = np.mean(pt_list)
    return {"fit": fit_times, "predict": pred_times}


def plot_results(
    results: Dict[str, np.ndarray],
    N_values: List[int],
    M_values: List[int],
    title: str,
) -> None:
    """Plot runtime as a function of N and M."""
    fit_times = results["fit"]
    pred_times = results["predict"]
    plt.figure(figsize=(8, 6))
    for idx, M in enumerate(M_values):
        plt.plot(N_values, fit_times[:, idx], marker="o", label=f"Fit: M={M}")
        plt.plot(N_values, pred_times[:, idx], marker="x", linestyle="--", label=f"Predict: M={M}")
    plt.xlabel("Number of samples (N)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    # Define ranges of N and M to test
    # Use smaller values by default for faster demonstration. Adjust as needed.
    N_values = [20, 40, 80]
    M_values = [5, 10]
    num_runs = 2  # Fewer runs for brevity; increase for more stable results

    # Case 1: Real input, real output (regression)
    res_rr = measure_time(N_values, M_values, input_type="real", output_type="real", num_runs=num_runs)
    plot_results(res_rr, N_values, M_values, "Real input, Real output")

    # Case 2: Real input, discrete output (classification)
    res_rc = measure_time(N_values, M_values, input_type="real", output_type="discrete", num_runs=num_runs)
    plot_results(res_rc, N_values, M_values, "Real input, Discrete output")

    # Case 3: Discrete input, real output (regression)
    res_cr = measure_time(N_values, M_values, input_type="discrete", output_type="real", num_runs=num_runs)
    plot_results(res_cr, N_values, M_values, "Discrete input, Real output")

    # Case 4: Discrete input, discrete output (classification)
    res_cc = measure_time(N_values, M_values, input_type="discrete", output_type="discrete", num_runs=num_runs)
    plot_results(res_cc, N_values, M_values, "Discrete input, Discrete output")


if __name__ == "__main__":
    main()