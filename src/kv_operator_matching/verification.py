"""
Verification gate for compact representations.

After fitting a candidate compact representation, verification checks
whether it meets quality thresholds before activation. This is the
service-safe gating step: expensive fitting happens at checkpoints,
verification gates deployment.

The verification protocol:
  1. Split the query bank into fit queries (used during beta-refit)
     and holdout queries (reserved for verification).
  2. Evaluate L_true (true response error) on the holdout set.
  3. Compare to the configured threshold.
  4. Return a VerificationResult indicating pass/fail and the metric value.

Using a held-out split prevents overfitting to the query bank: a
representation that perfectly matches the fit queries but fails on
holdout queries should not be deployed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import VerificationConfig
from .objectives import loss_true_response
from .query_bank import QueryBank
from .types import CompactRepresentation, HeadState


@dataclass
class VerificationResult:
    """Result of a verification check on a compact representation.

    Attributes:
        passed: True if the metric value is below the threshold.
        metric_value: The computed metric value on the holdout set.
        threshold: The threshold used for gating.
        metric_name: The name of the metric evaluated (e.g., "response_l2").
    """

    passed: bool
    metric_value: float
    threshold: float
    metric_name: str

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"VerificationResult({status}: {self.metric_name}={self.metric_value:.4f}, "
            f"threshold={self.threshold:.4f})"
        )


def verify(
    compact_rep: CompactRepresentation,
    ref_head_state: HeadState,
    query_bank: QueryBank,
    config: VerificationConfig,
) -> VerificationResult:
    """Evaluate a compact representation against a quality threshold.

    Splits the query bank into fit and holdout portions. Evaluates L_true
    (mean squared response error per query) on the holdout portion.

    Args:
        compact_rep: Candidate compact representation to verify.
        ref_head_state: Full KV cache state for the reference head.
        query_bank: Empirical query bank (will be split into holdout).
        config: VerificationConfig with threshold and holdout_fraction.

    Returns:
        VerificationResult with pass/fail status and metric value.

    Note:
        Currently only "response_l2" metric is implemented. The metric
        is the mean per-query L2 response error (normalized by the number
        of holdout queries), which makes it comparable across different
        bank sizes.
    """
    if config.metric != "response_l2":
        raise NotImplementedError(
            f"Metric '{config.metric}' is not yet implemented. "
            "Only 'response_l2' is supported."
        )

    _fit_bank, holdout_bank = query_bank.split_train_holdout(
        train_fraction=1.0 - config.holdout_fraction
    )
    holdout_queries, holdout_weights = holdout_bank.get_weighted_bank()

    with torch.no_grad():
        total_loss = loss_true_response(
            holdout_queries,
            holdout_weights,
            compact_rep.support_keys,
            compact_rep.support_values,
            compact_rep.betas,
            ref_head_state.keys,
            ref_head_state.values,
        )
        # Normalize by total weight to get a per-query average
        total_weight = holdout_weights.sum().clamp(min=1e-8)
        metric_value = (total_loss / total_weight).item()

    passed = metric_value <= config.threshold
    return VerificationResult(
        passed=passed,
        metric_value=metric_value,
        threshold=config.threshold,
        metric_name=config.metric,
    )
