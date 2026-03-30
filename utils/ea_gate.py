"""
Adaptive EA Hard Gating based on log-determinant of covariance matrix.

Uses the 1-sigma rule: if the target subject's log|det(R_bar)| falls below
mu - sigma (computed across all subjects), EA is disabled for the entire fold.

This avoids noise amplification on subjects with collapsed covariance manifolds.
"""
import numpy as np
from typing import Dict, List, Tuple


def _safe_covariance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute covariance of a single trial [C, T] -> [C, C]."""
    cov = (x @ x.T) / max(x.shape[1], 1)
    cov = 0.5 * (cov + cov.T)
    cov += eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return cov


def compute_log_det(x_train: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute log|det(R_bar)| for a subject's training data.

    Args:
        x_train: [N, C, T] training trials.
        eps: numerical stability.

    Returns:
        log determinant of the mean covariance matrix.
    """
    covs = [_safe_covariance(trial, eps=eps) for trial in x_train]
    ref_cov = np.mean(np.stack(covs, axis=0), axis=0)
    _, logdet = np.linalg.slogdet(ref_cov)
    return float(logdet)


def should_disable_ea(
    target_log_det: float,
    all_log_dets: List[float],
    n_sigma: float = 1.0,
) -> Tuple[bool, float, float, float]:
    """
    Determine whether EA should be disabled for the target subject.

    Uses the 1-sigma rule: disable if target_log_det < mu - n_sigma * sigma.

    Args:
        target_log_det: log|det| of the target subject.
        all_log_dets: log|det| values for ALL subjects (including target).
        n_sigma: number of standard deviations below mean to trigger gating.

    Returns:
        (gated, threshold, mu, sigma)
    """
    mu = float(np.mean(all_log_dets))
    sigma = float(np.std(all_log_dets, ddof=0))
    threshold = mu - n_sigma * sigma
    gated = target_log_det < threshold
    return gated, threshold, mu, sigma
