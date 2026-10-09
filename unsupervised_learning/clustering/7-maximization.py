#!/usr/bin/env python3

import numpy as np


def maximization(X, g):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None
    Nk = np.sum(g, axis=1)
    if np.any(Nk <= 0):
        return None, None, None
    pi = Nk / n
    m = np.matmul(g, X) / Nk[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul((g[i, :, None] * diff).T, diff) / Nk[i]
    return pi, m, S
