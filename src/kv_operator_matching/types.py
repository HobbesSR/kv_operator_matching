"""
Core data types for the kv_operator_matching library.

All types are dataclasses with typed fields. Validation is not yet
implemented — inputs are trusted to be well-formed.

TODO: add validation (shape checks, dtype checks, device consistency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class HeadState:
    """KV cache state for a single attention head at a single layer.

    Attributes:
        head_idx: Index of the attention head within the layer.
        layer_idx: Index of the transformer layer.
        keys: Key tensor of shape (seq_len, d_k).
        values: Value tensor of shape (seq_len, d_v).
    """

    head_idx: int
    layer_idx: int
    keys: Tensor
    values: Tensor


@dataclass
class QueryBank:
    """Empirical query bank for operator evidence collection.

    Stores a collection of query vectors and associated importance
    weights drawn from live inference. These define the empirical
    distribution over which the operator-matching objective is evaluated.

    Attributes:
        queries: Query tensor of shape (n_queries, d_k).
        weights: Importance weight tensor of shape (n_queries,).
            Should be nonneg; need not sum to 1 (normalized on use).
        metadata: Optional metadata dict (e.g., source layer/head,
            collection step, weighting scheme).
    """

    queries: Tensor
    weights: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactRepresentation:
    """A compact measure approximating a KV cache state.

    Represents the approximation mu_hat = sum_j beta_j * delta_{(k_hat_j, v_hat_j)}.
    Support points (support_keys, support_values) may or may not be drawn
    from the original KV cache — they can be synthetic or merged.

    Attributes:
        support_keys: Key support tensor of shape (m, d_k).
        support_values: Value support tensor of shape (m, d_v).
        betas: Nonneg coefficient tensor of shape (m,).
            Default (ones) corresponds to uniform weighting of support points.
    """

    support_keys: Tensor
    support_values: Tensor
    betas: Tensor


@dataclass
class OperatorStats:
    """Computed N/Z statistics for a set of queries.

    Stores the per-query partition function values (z_vals) and
    value numerator vectors (n_vals), along with the queries used.
    Useful for caching intermediate computations during beta-fit
    and verification.

    Attributes:
        z_vals: Partition function values, shape (n_queries,).
        n_vals: Value numerator vectors, shape (n_queries, d_v).
        queries: Query vectors used, shape (n_queries, d_k).
    """

    z_vals: Tensor
    n_vals: Tensor
    queries: Tensor
