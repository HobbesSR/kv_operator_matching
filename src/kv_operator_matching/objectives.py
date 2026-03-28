"""
N/Z operator matching objectives.

Defines the empirical surrogate losses (L_Z, L_N, L_lin) and the
true response error (L_true) used for verification.

Math notation (ASCII):
  Z_mu(q)  = sum_i beta_i * exp(<q, k_i>)
  N_mu(q)  = sum_i beta_i * exp(<q, k_i>) * v_i
  A_mu(q)  = N_mu(q) / Z_mu(q)

For the reference (full) cache, beta_i = 1 for all i.
For the compact representation, beta_j are fit coefficients.

Surrogate objectives (all functions of beta for fixed support):
  L_Z   = sum_t w_t * (Z_hat(q_t) - Z_ref(q_t))^2
  L_N   = sum_t w_t * ||N_hat(q_t) - N_ref(q_t)||^2
  L_lin = L_Z + (1/d_v) * L_N  (normalized; see loss_lin docstring)

Note on normalization: L_Z is scalar per query; L_N accumulates squared
error over d_v value dimensions. Without normalization, L_N dominates by
a factor of d_v (typically 64-128), making L_lin essentially ignore L_Z.
The default normalized form L_Z + L_N/d_v equalizes their contribution.

True response error:
  L_true = sum_t w_t * ||A_hat(q_t) - A_ref(q_t)||^2

L_true is non-convex in beta in general. It is used for verification
only, not for fitting.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def compute_z(
    queries: Tensor,
    keys: Tensor,
    betas: Optional[Tensor] = None,
) -> Tensor:
    """Compute the partition function Z_mu(q) for each query.

    Z_mu(q) = sum_i beta_i * exp(<q, k_i>)

    For numerical stability, we compute in log-space:
      log_z_i(q) = <q, k_i>  (no normalization applied here)
      Z_mu(q)    = sum_i beta_i * exp(<q, k_i>)

    To avoid overflow, we subtract the per-query max before exponentiating
    and restore the scale. This gives the same Z values as the raw formula
    when betas are all 1, but is stable for large inner products.

    Args:
        queries: Tensor of shape (n_queries, d_k).
        keys: Tensor of shape (n_keys, d_k).
        betas: Optional tensor of shape (n_keys,) with nonneg weights.
            If None, defaults to ones (unweighted sum).

    Returns:
        Tensor of shape (n_queries,) with Z_mu(q) for each query.
    """
    # logits[q, i] = <q_q, k_i>
    logits = queries @ keys.T  # (n_queries, n_keys)

    # Numerically stable: subtract per-query max
    max_logits = logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)  # (n_queries, n_keys)

    if betas is not None:
        exp_logits = exp_logits * betas.unsqueeze(0)  # broadcast over queries

    z = exp_logits.sum(dim=-1) * max_logits.squeeze(-1).exp()  # restore scale
    return z  # (n_queries,)


def compute_n(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    betas: Optional[Tensor] = None,
) -> Tensor:
    """Compute the value numerator N_mu(q) for each query.

    N_mu(q) = sum_i beta_i * exp(<q, k_i>) * v_i

    Args:
        queries: Tensor of shape (n_queries, d_k).
        keys: Tensor of shape (n_keys, d_k).
        values: Tensor of shape (n_keys, d_v).
        betas: Optional tensor of shape (n_keys,) with nonneg weights.
            If None, defaults to ones.

    Returns:
        Tensor of shape (n_queries, d_v) with N_mu(q) for each query.
    """
    logits = queries @ keys.T  # (n_queries, n_keys)
    max_logits = logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)  # (n_queries, n_keys)

    if betas is not None:
        exp_logits = exp_logits * betas.unsqueeze(0)

    # n[q] = sum_i exp_logits[q, i] * v_i
    n = exp_logits @ values  # (n_queries, d_v)
    # Restore scale (max_logits affects the magnitude uniformly per query)
    scale = max_logits.squeeze(-1).exp()  # (n_queries,)
    n = n * scale.unsqueeze(-1)
    return n  # (n_queries, d_v)


def compute_response(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    betas: Optional[Tensor] = None,
) -> Tensor:
    """Compute the attention response A_mu(q) = N_mu(q) / Z_mu(q).

    Args:
        queries: Tensor of shape (n_queries, d_k).
        keys: Tensor of shape (n_keys, d_k).
        values: Tensor of shape (n_keys, d_v).
        betas: Optional tensor of shape (n_keys,) with nonneg weights.

    Returns:
        Tensor of shape (n_queries, d_v) with A_mu(q) for each query.
    """
    logits = queries @ keys.T  # (n_queries, n_keys)
    max_logits = logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)  # (n_queries, n_keys)

    if betas is not None:
        exp_logits = exp_logits * betas.unsqueeze(0)

    z = exp_logits.sum(dim=-1, keepdim=True)  # (n_queries, 1)
    n = exp_logits @ values  # (n_queries, d_v)
    # Scale cancels: A = N / Z = (scale * n_stable) / (scale * z_stable)
    return n / z  # (n_queries, d_v)


def loss_z(
    queries: Tensor,
    weights: Tensor,
    keys_hat: Tensor,
    betas_hat: Tensor,
    keys_ref: Tensor,
    betas_ref: Optional[Tensor] = None,
) -> Tensor:
    """Compute the L_Z partition function surrogate loss.

    L_Z = sum_t w_t * (Z_hat(q_t) - Z_ref(q_t))^2

    Args:
        queries: Query bank tensor, shape (n_queries, d_k).
        weights: Importance weights, shape (n_queries,). Need not sum to 1.
        keys_hat: Support keys for compact representation, shape (m, d_k).
        betas_hat: Coefficients for compact representation, shape (m,).
        keys_ref: Keys for reference (full) cache, shape (n, d_k).
        betas_ref: Optional betas for reference cache, shape (n,).
            If None, defaults to ones.

    Returns:
        Scalar loss tensor.
    """
    z_hat = compute_z(queries, keys_hat, betas_hat)
    z_ref = compute_z(queries, keys_ref, betas_ref)
    sq_err = (z_hat - z_ref) ** 2  # (n_queries,)
    return (weights * sq_err).sum()


def loss_n(
    queries: Tensor,
    weights: Tensor,
    keys_hat: Tensor,
    values_hat: Tensor,
    betas_hat: Tensor,
    keys_ref: Tensor,
    values_ref: Tensor,
    betas_ref: Optional[Tensor] = None,
) -> Tensor:
    """Compute the L_N value numerator surrogate loss.

    L_N = sum_t w_t * ||N_hat(q_t) - N_ref(q_t)||_2^2

    Args:
        queries: Query bank tensor, shape (n_queries, d_k).
        weights: Importance weights, shape (n_queries,).
        keys_hat: Support keys for compact representation, shape (m, d_k).
        values_hat: Support values for compact representation, shape (m, d_v).
        betas_hat: Coefficients for compact representation, shape (m,).
        keys_ref: Keys for reference cache, shape (n, d_k).
        values_ref: Values for reference cache, shape (n, d_v).
        betas_ref: Optional betas for reference cache.

    Returns:
        Scalar loss tensor.
    """
    n_hat = compute_n(queries, keys_hat, values_hat, betas_hat)
    n_ref = compute_n(queries, keys_ref, values_ref, betas_ref)
    sq_err = ((n_hat - n_ref) ** 2).sum(dim=-1)  # (n_queries,)
    return (weights * sq_err).sum()


def loss_lin(
    queries: Tensor,
    weights: Tensor,
    keys_hat: Tensor,
    values_hat: Tensor,
    betas_hat: Tensor,
    keys_ref: Tensor,
    values_ref: Tensor,
    betas_ref: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tensor:
    """Compute the combined linear surrogate L_lin = L_Z + (1/d_v) * L_N.

    This is the primary fitting objective for fixed-support beta-refit.
    It is convex and quadratic in betas_hat for fixed support (keys_hat,
    values_hat), making it solvable as an NNLS problem.

    Normalization: L_Z is scalar-valued per query; L_N accumulates squared
    error over d_v value dimensions. Without normalization, L_N dominates by
    a factor of d_v (typically 64-128). When normalize=True (default), L_N
    is divided by d_v so that both terms contribute at comparable scale.
    Set normalize=False only for debugging or if you want the raw sum.

    Args:
        queries: Query bank tensor, shape (n_queries, d_k).
        weights: Importance weights, shape (n_queries,).
        keys_hat: Support keys for compact representation, shape (m, d_k).
        values_hat: Support values for compact representation, shape (m, d_v).
        betas_hat: Coefficients for compact representation, shape (m,).
        keys_ref: Keys for reference cache, shape (n, d_k).
        values_ref: Values for reference cache, shape (n, d_v).
        betas_ref: Optional betas for reference cache.
        normalize: If True (default), divide L_N by d_v before summing.

    Returns:
        Scalar loss tensor equal to L_Z + L_N (optionally normalized).
    """
    lz = loss_z(queries, weights, keys_hat, betas_hat, keys_ref, betas_ref)
    ln = loss_n(
        queries, weights, keys_hat, values_hat, betas_hat, keys_ref, values_ref, betas_ref
    )
    if normalize:
        d_v = values_hat.shape[-1]
        ln = ln / d_v
    return lz + ln


def loss_true_response(
    queries: Tensor,
    weights: Tensor,
    keys_hat: Tensor,
    values_hat: Tensor,
    betas_hat: Tensor,
    keys_ref: Tensor,
    values_ref: Tensor,
    betas_ref: Optional[Tensor] = None,
) -> Tensor:
    """Compute the true response error L_true = sum_t w_t ||A_hat(q_t) - A_ref(q_t)||^2.

    This is the gold-standard evaluation metric. It is non-convex in
    betas_hat and should be used for verification only, not for fitting.

    Args:
        queries: Query bank tensor, shape (n_queries, d_k).
        weights: Importance weights, shape (n_queries,).
        keys_hat: Support keys for compact representation, shape (m, d_k).
        values_hat: Support values for compact representation, shape (m, d_v).
        betas_hat: Coefficients for compact representation, shape (m,).
        keys_ref: Keys for reference cache, shape (n, d_k).
        values_ref: Values for reference cache, shape (n, d_v).
        betas_ref: Optional betas for reference cache.

    Returns:
        Scalar loss tensor.
    """
    a_hat = compute_response(queries, keys_hat, values_hat, betas_hat)
    a_ref = compute_response(queries, keys_ref, values_ref, betas_ref)
    sq_err = ((a_hat - a_ref) ** 2).sum(dim=-1)  # (n_queries,)
    return (weights * sq_err).sum()
