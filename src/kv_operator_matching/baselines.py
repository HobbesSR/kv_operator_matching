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
