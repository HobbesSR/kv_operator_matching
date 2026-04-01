"""
Fixed-support value fitting for Phase 1b operator matching.

Implements a shared weighted least-squares family over a fixed support:
with support keys and betas fixed, solve for compact values that best match
the full-head response operator over the query bank.

In the current Phase 3C framing:
- `fit_values` / `refit_values` use the neutral row metric (`vfit`)
- `fit_values_quotient` / `refit_values_quotient` use the quotient row metric
  (`qvfit`)
- gated and tempered variants control how strongly the quotient-induced row
  scaling is applied
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import BetaFitConfig
from .objectives import compute_logits, compute_response
from .query_bank import QueryBank
from .types import CompactRepresentation


def _fit_value_matrix(
    design: Tensor,
    targets: Tensor,
    ridge: float,
) -> Tensor:
    """Solve weighted ridge least squares for compact values."""
    gram = design.T @ design
    rhs = design.T @ targets
    if ridge > 0:
        diag_mean = torch.clamp_min(torch.diagonal(gram).mean(), 1e-12)
        ridge_strength = ridge * diag_mean
        gram = gram + torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        ) * ridge_strength
    return torch.linalg.solve(gram, rhs)


def fit_values(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
) -> Tensor:
    """Fit compact values on a fixed support with fixed betas.

    Uses the compact attention weights as the design matrix and the full-head
    responses as the targets:

        min_V sum_t w_t ||A_hat(q_t; V) - A_ref(q_t)||^2

    where A_hat(q_t; V) = alpha_hat(q_t) @ V and alpha_hat is determined only
    by support_keys and betas.
    """
    queries, weights = query_bank.get_weighted_bank()
    queries = queries.float()
    weights = weights.float()
    support_keys = support_keys.float()
    betas = betas.float()
    ref_keys = ref_keys.float()
    ref_values = ref_values.float()

    with torch.no_grad():
        logits_hat = compute_logits(queries, support_keys)
        log_betas = torch.log(betas.clamp(min=1e-30)).unsqueeze(0)
        attn_weights = torch.softmax(logits_hat + log_betas, dim=-1)

        targets = compute_response(queries, ref_keys, ref_values)
        sqrt_w = weights.sqrt().unsqueeze(-1)
        design = attn_weights * sqrt_w
        weighted_targets = targets * sqrt_w

        return _fit_value_matrix(design, weighted_targets, ridge=config.value_ridge)


def fit_values_quotient(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
    *,
    row_scale_power: float = 1.0,
) -> Tensor:
    """Fit compact values using the fixed-support quotient-residual objective.

    With support keys and betas fixed, z_hat(q) is fixed. The exact local
    quotient residual is then:

        E(q; V) = N_hat(q; V) - A_ref(q) * Z_hat(q)

    which is linear in V. This solve therefore minimizes:

        min_V sum_t w_t ||E(q_t; V)||^2

    using the same anchored interpolation scheme as ordinary value fitting.

    This is still a local bank objective. It does not remove the need to judge
    concatenation behavior on held-out response error.
    """
    queries, weights = query_bank.get_weighted_bank()
    queries = queries.float()
    weights = weights.float()
    support_keys = support_keys.float()
    betas = betas.float()
    ref_keys = ref_keys.float()
    ref_values = ref_values.float()

    if row_scale_power < 0.0:
        raise ValueError(f"row_scale_power must be nonnegative, got {row_scale_power!r}.")

    with torch.no_grad():
        logits_hat = compute_logits(queries, support_keys)
        logits_ref = compute_logits(queries, ref_keys)
        shared_max = torch.maximum(
            logits_hat.max(dim=-1).values,
            logits_ref.max(dim=-1).values,
        ).unsqueeze(-1)

        exp_hat = torch.exp(logits_hat - shared_max) * betas.unsqueeze(0)
        z_hat = exp_hat.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        alpha_hat = exp_hat / z_hat
        row_scale = z_hat.pow(row_scale_power)
        targets = compute_response(queries, ref_keys, ref_values) * row_scale

        sqrt_w = weights.sqrt().unsqueeze(-1)
        design = alpha_hat * row_scale * sqrt_w
        weighted_targets = targets * sqrt_w

        return _fit_value_matrix(design, weighted_targets, ridge=config.value_ridge)


def compute_qvfit_row_scaling_stats(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    query_bank: QueryBank,
) -> dict[str, float]:
    """Return quotient-row-scaling diagnostics for fixed-support qvfit."""
    queries, weights = query_bank.get_weighted_bank()
    queries = queries.float()
    weights = weights.float()
    support_keys = support_keys.float()
    betas = betas.float()
    ref_keys = ref_keys.float()

    with torch.no_grad():
        logits_hat = compute_logits(queries, support_keys)
        logits_ref = compute_logits(queries, ref_keys)
        shared_max = torch.maximum(
            logits_hat.max(dim=-1).values,
            logits_ref.max(dim=-1).values,
        ).unsqueeze(-1)

        exp_hat = torch.exp(logits_hat - shared_max) * betas.unsqueeze(0)
        exp_ref = torch.exp(logits_ref - shared_max)
        z_hat = exp_hat.sum(dim=-1).clamp(min=1e-30)
        z_ref = exp_ref.sum(dim=-1).clamp(min=1e-30)
        z_ratio = z_hat / z_ref
        weight_total = weights.sum().clamp(min=1e-12)
        z_ratio_mean = float((weights * z_ratio).sum().item() / weight_total.item())
        z_ratio_centered = z_ratio - z_ratio_mean
        z_ratio_var = float((weights * z_ratio_centered.square()).sum().item() / weight_total.item())
        row_energy = weights * z_hat.square()
        row_energy_total = row_energy.sum().clamp(min=1e-12)
        topk = min(5, int(row_energy.numel()))
        row_energy_top5_share = float(row_energy.topk(topk).values.sum().item() / row_energy_total.item())
        row_mean = float(row_energy.mean().item())
        row_std = float(row_energy.std(unbiased=False).item())
        row_energy_cv = row_std / max(row_mean, 1e-12)
        return {
            "zhat_over_zref_mean": z_ratio_mean,
            "zhat_over_zref_cv": z_ratio_var**0.5 / max(z_ratio_mean, 1e-12),
            "q_row_energy_top5_share": row_energy_top5_share,
            "q_row_energy_cv": row_energy_cv,
        }


def refit_values(
    compact_rep: CompactRepresentation,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
) -> CompactRepresentation:
    """Fit values while keeping support keys and betas fixed.

    The fitted values are interpolated with the original support values using
    config.value_interpolation so Phase 1b defaults to an anchored update
    rather than an unconstrained replacement.
    """
    fitted_values = fit_values(
        compact_rep.support_keys,
        compact_rep.betas,
        ref_keys,
        ref_values,
        query_bank,
        config,
    )
    alpha = float(config.value_interpolation)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"value_interpolation must be in [0, 1], got {config.value_interpolation!r}."
        )
    new_values = compact_rep.support_values * (1.0 - alpha) + fitted_values * alpha
    return CompactRepresentation(
        support_keys=compact_rep.support_keys,
        support_values=new_values,
        betas=compact_rep.betas,
    )


def refit_values_quotient(
    compact_rep: CompactRepresentation,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
    *,
    row_scale_power: float = 1.0,
) -> CompactRepresentation:
    """Refit values with the fixed-support quotient-residual objective."""
    fitted_values = fit_values_quotient(
        compact_rep.support_keys,
        compact_rep.betas,
        ref_keys,
        ref_values,
        query_bank,
        config,
        row_scale_power=row_scale_power,
    )
    alpha = float(config.value_interpolation)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"value_interpolation must be in [0, 1], got {config.value_interpolation!r}."
        )
    new_values = compact_rep.support_values * (1.0 - alpha) + fitted_values * alpha
    return CompactRepresentation(
        support_keys=compact_rep.support_keys,
        support_values=new_values,
        betas=compact_rep.betas,
    )


def refit_values_quotient_gated(
    compact_rep: CompactRepresentation,
    ref_keys: Tensor,
    ref_values: Tensor,
    query_bank: QueryBank,
    config: BetaFitConfig,
    *,
    zhat_over_zref_cv_threshold: float,
) -> CompactRepresentation:
    """Use qvfit only when quotient row-scaling dispersion is sufficiently tame."""
    stats = compute_qvfit_row_scaling_stats(
        compact_rep.support_keys,
        compact_rep.betas,
        ref_keys,
        query_bank,
    )
    if stats["zhat_over_zref_cv"] <= zhat_over_zref_cv_threshold:
        return refit_values_quotient(
            compact_rep,
            ref_keys,
            ref_values,
            query_bank,
            config,
        )
    return refit_values(
        compact_rep,
        ref_keys,
        ref_values,
        query_bank,
        config,
    )
