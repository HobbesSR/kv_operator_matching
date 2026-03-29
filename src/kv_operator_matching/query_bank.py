"""
Empirical query bank for operator evidence collection.

During inference, live query vectors are collected and maintained in a
rolling bank that forms the basis for the empirical operator-matching
objective.

The bank supports recency-weighted decay of older queries and enforces
a maximum size by evicting the oldest entries when full.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .config import QueryBankConfig


class QueryBank:
    """Rolling bank of empirical query vectors with importance weights.

    Maintains a collection of query vectors drawn from live inference.
    When the bank reaches max_queries, the oldest entries are evicted.
    Weights are updated according to the configured weighting scheme.

    Usage::

        bank = QueryBank(QueryBankConfig(max_queries=256, weighting_scheme="recency"))
        bank.add_queries(q_tensor)  # shape (n, d_k)
        queries, weights = bank.get_weighted_bank()

    TODO: add support for attention-mass weighting once live attention
    evidence is available during inference.
    """

    def __init__(self, config: QueryBankConfig) -> None:
        """Initialize an empty query bank.

        Args:
            config: QueryBankConfig controlling capacity and weighting.
        """
        self.config = config
        self._queries: Optional[Tensor] = None
        self._weights: Optional[Tensor] = None

    def add_queries(self, queries: Tensor, weights: Optional[Tensor] = None) -> None:
        """Add new query vectors to the bank.

        If the bank exceeds max_queries after adding, the oldest entries
        are evicted to bring it back to max_queries.

        For "recency" weighting, existing weights are multiplied by
        decay_factor before the new entries are appended.

        Args:
            queries: Tensor of shape (n, d_k) containing new query vectors.
            weights: Optional tensor of shape (n,) with importance weights
                for the new queries. If None, defaults to ones.
        """
        if queries.dim() != 2:
            raise ValueError(f"queries must be 2D (n, d_k), got shape {queries.shape}")

        n = queries.shape[0]
        if weights is None:
            weights = torch.ones(n, dtype=queries.dtype, device=queries.device)

        if weights.shape != (n,):
            raise ValueError(
                f"weights must have shape ({n},), got {weights.shape}"
            )

        # Apply recency decay to existing entries
        if self._weights is not None and self.config.weighting_scheme == "recency":
            self._weights = self._weights * self.config.decay_factor

        # Concatenate new queries and weights
        if self._queries is None:
            self._queries = queries
            self._weights = weights
        else:
            self._queries = torch.cat([self._queries, queries], dim=0)
            self._weights = torch.cat([self._weights, weights], dim=0)

        # Evict oldest entries if over capacity
        if self._queries.shape[0] > self.config.max_queries:
            excess = self._queries.shape[0] - self.config.max_queries
            self._queries = self._queries[excess:]
            self._weights = self._weights[excess:]

    def get_weighted_bank(self) -> tuple[Tensor, Tensor]:
        """Return the current (queries, weights) pair.

        Weights are returned as-is (not renormalized). Callers that need
        normalized weights should divide by weights.sum().

        Returns:
            Tuple of (queries, weights) where queries has shape (n, d_k)
            and weights has shape (n,).

        Raises:
            RuntimeError: If the bank is empty.
        """
        if self._queries is None or self._weights is None:
            raise RuntimeError("Query bank is empty. Call add_queries() first.")
        return self._queries, self._weights

    def split_train_holdout(self, train_fraction: float = 0.5) -> tuple["QueryBank", "QueryBank"]:
        """Split the bank into deterministic train and holdout banks.

        The split preserves order and interleaves assignments so both splits
        cover the bank rather than assigning one contiguous suffix to holdout.
        This avoids recency-only leakage in prefill-only experiments.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction!r}.")

        queries, weights = self.get_weighted_bank()
        train_indices = []
        holdout_indices = []
        accumulator = 0.0
        for index in range(queries.shape[0]):
            accumulator += train_fraction
            if accumulator >= 1.0:
                train_indices.append(index)
                accumulator -= 1.0
            else:
                holdout_indices.append(index)

        if not train_indices and holdout_indices:
            train_indices.append(holdout_indices.pop(0))
        if not holdout_indices and train_indices:
            holdout_indices.append(train_indices[-1])

        train_bank = QueryBank(self.config)
        holdout_bank = QueryBank(self.config)
        train_bank.add_queries(queries[train_indices], weights[train_indices])
        holdout_bank.add_queries(queries[holdout_indices], weights[holdout_indices])
        return train_bank, holdout_bank

    def __len__(self) -> int:
        """Return the number of queries currently in the bank."""
        if self._queries is None:
            return 0
        return self._queries.shape[0]

    def reset(self) -> None:
        """Clear all queries and weights from the bank."""
        self._queries = None
        self._weights = None
