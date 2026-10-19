"""Helpers for loading data in the binary format FastSRS expects."""

import numpy as np
import pandas as pd


def load_binary_csv(path, target_col="Class", drop_constant=True):
    """Load an all-binary CSV into the ``(X, y, col_ls)`` triple used by ``ORS``.

    Parameters
    ----------
    path : str or path-like
        CSV with 0/1 indicator columns and an integer target column. Use
        :func:`fastsrs.preprocess_data` to build one from raw data.
    target_col : str, default ``"Class"``
        Name of the target column.
    drop_constant : bool, default ``True``
        Remove feature columns that take a single value, which carry no
        information and would otherwise slow down rule mining.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_features)
    y : numpy.ndarray of shape (n_samples,), dtype float
    col_ls : numpy.ndarray of feature names aligned with the columns of ``X``
    """
    df = pd.read_csv(path, header=0, sep=",")
    if target_col not in df.columns:
        raise ValueError(
            f"target column {target_col!r} not found in {path}; "
            f"available columns: {list(df.columns)[:10]}..."
        )
    y = np.array(df[target_col], dtype=float)
    df = df.drop(target_col, axis=1)
    col_ls = df.columns.values.copy()
    X = df.to_numpy()

    if drop_constant:
        constant = [c for c in range(X.shape[1]) if len(np.unique(X[:, c])) == 1]
        X = np.delete(X, constant, axis=1)
        col_ls = np.delete(col_ls, constant)

    return X, y, col_ls
