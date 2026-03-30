from __future__ import annotations

import numpy as np


def _inv_sqrtm_spd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute inverse square-root of an SPD matrix with eigenvalue clipping."""
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.clip(evals, eps, None)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return inv_sqrt.astype(np.float32)


def fit_ea_reference(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Fit Euclidean Alignment (EA) reference from EEG trials.

    Args:
        X: EEG trials with shape [n_trials, n_channels, n_times].
        eps: Numerical floor for stable matrix inverse square-root.

    Returns:
        Inverse square-root of mean covariance, shape [n_channels, n_channels].
    """
    if X.ndim != 3:
        raise ValueError(f"EA expects 3D input [N, C, T], got shape {X.shape}")

    covariances = np.matmul(X, np.transpose(X, (0, 2, 1)))
    covariances /= max(X.shape[-1], 1)
    mean_cov = np.mean(covariances, axis=0)
    return _inv_sqrtm_spd(mean_cov, eps=eps)


def apply_ea(X: np.ndarray, reference_inv_sqrt: np.ndarray) -> np.ndarray:
    """Apply Euclidean Alignment to EEG trials [N, C, T]."""
    if X.ndim != 3:
        raise ValueError(f"EA expects 3D input [N, C, T], got shape {X.shape}")
    aligned = np.einsum("ij,njt->nit", reference_inv_sqrt, X)
    return aligned.astype(np.float32)
