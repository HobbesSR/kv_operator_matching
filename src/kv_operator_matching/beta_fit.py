"""
Beta coefficient fitting for fixed-support operator matching.

Given a fixed support (k_hat_j, v_hat_j), fits nonneg coefficients beta_j to
minimize the empirical surrogate objective over the query bank.

The L_lin objective (L_Z + L_N) is convex quadratic in beta for fixed support,
making this a nonneg least squares (NNLS) problem. The current implementation
uses scipy.optimize.nnls as a placeholder. A more efficient implementation
using the structure of the problem (block-diagonal Gram matrix, etc.) is left
as a TODO.

TODO: implement a proper batched NNLS that handles the Z and N blocks jointly
      without constructing the full feature matrix explicitly (can be expensive
      for large support sizes and query banks).
TODO: consider an active-set or projected gradient implementation in PyTorch
      to stay on-device and avoid CPU-GPU copies.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .config import BetaFitConfig
from .query_bank import QueryBank
from .types import CompactRepresentation


def fit_beta(
    support_keys: Tensor,
    support_values: Tensor,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
) -> Tensor:
    """Fit nonneg beta coefficients for a fixed support to minimize L_lin.

    Given fixed support points (support_keys, support_values) and a reference
    KV cache (ref_keys, ref_values), solves:

        min_{beta >= 0} L_lin(beta)
            = sum_t w_t * (Z_hat(q_t) - Z_ref(q_t))^2
            + sum_t w_t * ||N_hat(q_t) - N_ref(q_t)||^2

    where Z_hat(q) = sum_j beta_j * exp(<q, k_hat_j>) and similarly for N_hat.

    Since L_lin is quadratic in beta, this reduces to an NNLS problem:

        min_{beta >= 0} ||Phi @ beta - y||^2

    where Phi is the weighted feature matrix constructed from the query bank
    and the support key/value vectors, and y is the reference target vector.

    Currently implemented via scipy.optimize.nnls (CPU, single-threaded).
    Inputs are moved to CPU for the solve and the result is returned as a
    CPU tensor. Caller is responsible for moving to the appropriate device.

    TODO: Implement proper NNLS or constrained optimization using a
          PyTorch-native solver to avoid device transfer overhead.

    Args:
        support_keys: Fixed support keys, shape (m, d_k).
        support_values: Fixed support values, shape (m, d_v).
        ref_keys: Reference KV cache keys, shape (n, d_k).
        ref_values: Reference KV cache values, shape (n, d_v).
        query_bank: Empirical query bank (provides queries and weights).
        config: BetaFitConfig controlling solver options.

    Returns:
        Tensor of shape (m,) with nonneg beta coefficients.
    """
    from scipy.optimize import nnls  # lazy import: scipy may not always be needed

    queries, weights = query_bank.get_weighted_bank()
    queries = queries.float().cpu()
    weights = weights.float().cpu()
    support_keys_cpu = support_keys.float().cpu()
    support_values_cpu = support_values.float().cpu()
    ref_keys_cpu = ref_keys.float().cpu()
    ref_values_cpu = ref_values.float().cpu()

    n_queries = queries.shape[0]
    m = support_keys_cpu.shape[0]
    d_v = support_values_cpu.shape[1]

    # Build feature matrix Phi and target vector y for the NNLS problem.
    #
    # The L_Z block:
    #   Z_hat(q_t) = sum_j beta_j * exp(<q_t, k_hat_j>)
    #   Feature row for query t: sqrt(w_t) * [exp(<q_t, k_hat_j>)]_j  (shape: m)
    #   Target for query t: sqrt(w_t) * Z_ref(q_t)
    #
    # The L_N block (vectorized over value dims):
    #   N_hat(q_t)[d] = sum_j beta_j * exp(<q_t, k_hat_j>) * v_hat_j[d]
    #   Feature row for query t, value dim d:
    #     sqrt(w_t) * [exp(<q_t, k_hat_j>) * v_hat_j[d]]_j  (shape: m)
    #   Target for query t, value dim d: sqrt(w_t) * N_ref(q_t)[d]
    #
    # Total rows: n_queries * (1 + d_v), columns: m

    with torch.no_grad():
        # Compute exp(<q, k_hat>) for all queries and support points
        logits_hat = queries @ support_keys_cpu.T  # (n_queries, m)
        max_logits_hat = logits_hat.max(dim=-1, keepdim=True).values
        exp_hat = torch.exp(logits_hat - max_logits_hat)  # (n_queries, m)
        scale_hat = max_logits_hat.squeeze(-1).exp()  # (n_queries,)
        exp_hat_scaled = exp_hat * scale_hat.unsqueeze(-1)  # (n_queries, m)

        # Reference Z and N
        logits_ref = queries @ ref_keys_cpu.T  # (n_queries, n_ref)
        max_logits_ref = logits_ref.max(dim=-1, keepdim=True).values
        exp_ref = torch.exp(logits_ref - max_logits_ref)
        scale_ref = max_logits_ref.squeeze(-1).exp()
        exp_ref_scaled = exp_ref * scale_ref.unsqueeze(-1)
        z_ref = exp_ref_scaled.sum(dim=-1)  # (n_queries,)
        n_ref = exp_ref_scaled @ ref_values_cpu  # (n_queries, d_v)

        sqrt_w = weights.sqrt()  # (n_queries,)

        # Z block: shape (n_queries, m)
        phi_z = (exp_hat_scaled * sqrt_w.unsqueeze(-1))  # (n_queries, m)
        y_z = (z_ref * sqrt_w)  # (n_queries,)

        # N block: shape (n_queries * d_v, m)
        # phi_n[q * d_v + d, j] = sqrt(w_q) * exp_hat_scaled[q, j] * v_hat_j[d]
        # = phi_z[q, j] * v_hat_j[d]
        # Vectorize: (n_queries, m, 1) * (1, m, d_v) -> (n_queries, m, d_v)
        phi_n_3d = phi_z.unsqueeze(-1) * support_values_cpu.T.unsqueeze(0)
        # -> (n_queries, m, d_v); reshape to (n_queries * d_v, m)
        phi_n = phi_n_3d.permute(0, 2, 1).reshape(n_queries * d_v, m)
        y_n = (n_ref * sqrt_w.unsqueeze(-1)).reshape(n_queries * d_v)

        # Stack Z and N blocks
        Phi = torch.cat([phi_z, phi_n], dim=0).numpy()  # ((1+d_v)*n_queries, m)
        y = torch.cat([y_z, y_n], dim=0).numpy()  # ((1+d_v)*n_queries,)

    beta_np, _residual = nnls(Phi, y, maxiter=config.max_iter)
    return torch.tensor(beta_np, dtype=torch.float32)


def refit_beta(
    compact_rep: CompactRepresentation,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
) -> CompactRepresentation:
    """Convenience wrapper: refit beta on an existing CompactRepresentation.

    Keeps the support (keys and values) fixed; replaces betas with the
    newly fitted coefficients.

    Args:
        compact_rep: Existing compact representation (support is kept fixed).
        ref_keys: Reference KV cache keys, shape (n, d_k).
        ref_values: Reference KV cache values, shape (n, d_v).
        query_bank: Empirical query bank.
        config: BetaFitConfig controlling solver options.

    Returns:
        A new CompactRepresentation with the same support but updated betas.
    """
    new_betas = fit_beta(
        compact_rep.support_keys,
        compact_rep.support_values,
        ref_keys,
        ref_values,
        query_bank,
        config,
    )
    return CompactRepresentation(
        support_keys=compact_rep.support_keys,
        support_values=compact_rep.support_values,
        betas=new_betas,
    )
