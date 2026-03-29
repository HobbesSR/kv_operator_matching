"""
Baseline support selection methods for comparison.

These implement the standard baselines from the prior kv_compaction_experiment
and equivalent methods, adapted to return CompactRepresentation objects
compatible with the N/Z objective framework.

All baselines return a CompactRepresentation with betas=ones by default
(no beta-refit). To get the beta-refit variant, pass the returned
CompactRepresentation to beta_fit.refit_beta() with the reference KV state
and a query bank.

TODO (for each baseline): wire up beta-refit in the experiment pipeline
     and compare no-refit vs. refit variants.
"""
from __future__ import annotations

import math

import torch

from .objectives import compute_logits
from .query_bank import QueryBank
from .types import CompactRepresentation, HeadState


def recency_baseline(head_state: HeadState, budget: int) -> CompactRepresentation:
    """Select the most recent `budget` KV pairs as the compact representation.

    This is the simplest possible baseline: keep the last `budget` tokens
    in the KV cache and discard the rest. Motivated by the observation that
    recent tokens tend to receive higher attention mass than distant ones
    in many generation settings.

    Args:
        head_state: Full KV cache state for one head.
        budget: Number of support points to retain (m <= n).

    Returns:
        CompactRepresentation with the last `budget` KV pairs and betas=ones.
        If the cache has fewer than `budget` entries, all are returned.

    TODO: add optional beta-refit via BetaFitConfig argument.
    """
    keys = head_state.keys
    values = head_state.values
    n = keys.shape[0]
    m = min(budget, n)
    support_keys = keys[-m:]
    support_values = values[-m:]
    betas = torch.ones(m, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def attention_mass_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
) -> CompactRepresentation:
    """Select the `budget` KV pairs with highest total attention mass over the query bank.

    Attention mass for key i is defined as:
        mass_i = sum_t w_t * softmax(<q_t, k_i>)_i

    where the softmax is computed over all keys in the full cache for each
    query q_t. Entries with the highest total mass are selected as the support.

    This is the "attention mass" baseline from kv_compaction_experiment,
    adapted to use the N/Z query bank as the source of query evidence rather
    than a fixed offline query set.

    Args:
        head_state: Full KV cache state for one head.
        query_bank: Empirical query bank providing weighted queries.
        budget: Number of support points to retain (m <= n).

    Returns:
        CompactRepresentation with the top-mass KV pairs and betas=ones.

    TODO: add optional beta-refit via BetaFitConfig argument.
    TODO: consider using softmax of the query bank's own weights rather
          than recomputing full softmax for efficiency.
    """
    queries, weights = query_bank.get_weighted_bank()
    keys = head_state.keys
    values = head_state.values
    n = keys.shape[0]
    m = min(budget, n)

    queries_f = queries.float()
    keys_f = keys.float()
    weights_f = weights.float()

    # Compute softmax attention weights over full cache for each query.
    logits = compute_logits(queries_f, keys_f)  # (n_queries, n)
    attn_weights = torch.softmax(logits, dim=-1)  # (n_queries, n)

    # Aggregate: mass[i] = sum_t w_t * attn_weights[t, i]
    mass = (weights_f.unsqueeze(-1) * attn_weights).sum(dim=0)  # (n,)

    # Select top-m indices by mass
    top_indices = torch.topk(mass, k=m).indices

    support_keys = keys[top_indices]
    support_values = values[top_indices]
    betas = torch.ones(m, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def omp_mass_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
) -> CompactRepresentation:
    """Select support with local OMP over the query-conditioned mass frame.

    This is the closest local analogue to the paper's AM-OMP selector available
    in this repo today. It greedily selects keys to match the reference
    partition-function frame over the query bank, jointly producing a support
    and nonnegative beta coefficients.

    The support values are copied from the underlying KV cache; value fitting is
    a separate downstream step.
    """
    queries, weights = query_bank.get_weighted_bank()
    keys = head_state.keys
    values = head_state.values
    n = keys.shape[0]
    m = min(budget, n)
    if m <= 0:
        return CompactRepresentation(
            support_keys=keys[:0],
            support_values=values[:0],
            betas=torch.ones(0, dtype=keys.dtype, device=keys.device),
        )

    queries_f = queries.float()
    keys_f = keys.float()
    weights_f = weights.float()

    selected_indices, selected_betas = _select_keys_with_omp(
        key_tensor=keys_f,
        query_tensor=queries_f,
        entry_weights=weights_f,
        selection_budget=m,
    )
    if not selected_indices:
        return attention_mass_baseline(head_state, query_bank, budget)

    index_tensor = torch.tensor(selected_indices, device=keys.device, dtype=torch.long)
    support_keys = keys[index_tensor]
    support_values = values[index_tensor]
    betas = torch.tensor(selected_betas, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def _select_keys_with_omp(
    *,
    key_tensor: torch.Tensor,
    query_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
    selection_budget: int,
) -> tuple[list[int], list[float]]:
    """Greedy OMP-like support selection on the stable exp-logit mass frame."""
    if selection_budget <= 0 or key_tensor.numel() == 0 or query_tensor.numel() == 0:
        return [], []

    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    logits = (query_tensor @ key_tensor.T) * inv_sqrt_d
    reference_max = logits.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(logits - reference_max)
    target = exp_scores.sum(dim=1)
    row_weights = torch.sqrt(torch.clamp_min(entry_weights.to(dtype=torch.float32), 0.0))
    weighted_design = exp_scores * row_weights.unsqueeze(1)
    weighted_target = target * row_weights

    selected_indices: list[int] = []
    mask = torch.zeros(weighted_design.shape[1], dtype=torch.bool, device=weighted_design.device)
    current = torch.zeros_like(weighted_target)
    scale = None

    for _ in range(min(int(selection_budget), int(weighted_design.shape[1]))):
        residual = weighted_target - current
        corr = torch.matmul(weighted_design.T, residual)
        corr[mask] = -float("inf")
        index = int(torch.argmax(corr).item())
        if not math.isfinite(float(corr[index].item())):
            break
        selected_indices.append(index)
        mask[index] = True
        selected_design = weighted_design[:, selected_indices]
        scale = torch.linalg.lstsq(
            selected_design,
            weighted_target.unsqueeze(1),
            driver="gels",
        ).solution.squeeze(1)
        scale = scale.clamp_min(1e-12)
        current = selected_design @ scale

    if scale is None:
        return [], []
    return selected_indices, [float(value) for value in scale.tolist()]


def uniform_baseline(head_state: HeadState, budget: int) -> CompactRepresentation:
    """Select `budget` KV pairs uniformly at random (without replacement).

    Intended as a sanity-check baseline. Any method that does not beat
    uniform random selection is not useful. Results will vary across runs
    unless a fixed seed is set by the caller.

    Args:
        head_state: Full KV cache state for one head.
        budget: Number of support points to retain (m <= n).

    Returns:
        CompactRepresentation with uniformly sampled KV pairs and betas=ones.

    TODO: add optional beta-refit via BetaFitConfig argument.
    """
    keys = head_state.keys
    values = head_state.values
    n = keys.shape[0]
    m = min(budget, n)

    indices = torch.randperm(n, device=keys.device)[:m]
    support_keys = keys[indices]
    support_values = values[indices]
    betas = torch.ones(m, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )
