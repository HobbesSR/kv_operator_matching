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


def _compute_qvfit_row_terms(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    query_bank: QueryBank,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return bank weights plus fixed-support quotient row scales."""
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
    return weights, z_hat, z_ref


def _compute_qvfit_row_scaling_stats_from_terms(
    weights: Tensor,
    z_hat: Tensor,
    z_ref: Tensor,
    *,
    row_scale_power: float = 1.0,
) -> dict[str, float]:
    """Summarize the quotient-induced row metric at a chosen power."""
    if row_scale_power < 0.0:
        raise ValueError(f"row_scale_power must be nonnegative, got {row_scale_power!r}.")

    weight_total = weights.sum().clamp(min=1e-12)
    z_ratio = z_hat / z_ref
    z_ratio_mean = float((weights * z_ratio).sum().item() / weight_total.item())
    z_ratio_centered = z_ratio - z_ratio_mean
    z_ratio_var = float((weights * z_ratio_centered.square()).sum().item() / weight_total.item())

    neutral_probs = (weights / weight_total).clamp(min=1e-30)
    row_energy = weights * z_hat.pow(2.0 * row_scale_power)
    row_energy_total = row_energy.sum().clamp(min=1e-12)
    row_probs = (row_energy / row_energy_total).clamp(min=1e-30)

    topk = min(5, int(row_energy.numel()))
    row_energy_top5_share = float(row_energy.topk(topk).values.sum().item() / row_energy_total.item())
    row_mean = float(row_energy.mean().item())
    row_std = float(row_energy.std(unbiased=False).item())
    row_energy_cv = row_std / max(row_mean, 1e-12)
    q_weight_neff = float(1.0 / row_probs.square().sum().item())
    neutral_neff = float(1.0 / neutral_probs.square().sum().item())
    q_weight_kl_to_neutral = float(
        (row_probs * (torch.log(row_probs) - torch.log(neutral_probs))).sum().item()
    )
    q_weight_entropy = float((-(row_probs * torch.log(row_probs))).sum().item())
    return {
        "row_scale_power": float(row_scale_power),
        "zhat_over_zref_mean": z_ratio_mean,
        "zhat_over_zref_cv": z_ratio_var**0.5 / max(z_ratio_mean, 1e-12),
        "q_row_energy_top5_share": row_energy_top5_share,
        "q_row_energy_cv": row_energy_cv,
        "q_weight_neff": q_weight_neff,
        "q_weight_neff_fraction": q_weight_neff / max(neutral_neff, 1e-12),
        "q_weight_kl_to_neutral": q_weight_kl_to_neutral,
        "q_weight_entropy": q_weight_entropy,
    }


def compute_qvfit_row_scaling_stats(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    query_bank: QueryBank,
    *,
    row_scale_power: float = 1.0,
) -> dict[str, float]:
    """Return quotient-row-scaling diagnostics for fixed-support qvfit."""
    weights, z_hat, z_ref = _compute_qvfit_row_terms(
        support_keys,
        betas,
        ref_keys,
        query_bank,
    )
    return _compute_qvfit_row_scaling_stats_from_terms(
        weights,
        z_hat,
        z_ref,
        row_scale_power=row_scale_power,
    )


def choose_qvfit_row_scale_power(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    query_bank: QueryBank,
    *,
    min_neff_fraction: float | None = None,
    max_kl_to_neutral: float | None = None,
    grid_size: int = 65,
) -> tuple[float, dict[str, float]]:
    """Choose the strongest quotient row metric that stays bank-representative."""
    if min_neff_fraction is None and max_kl_to_neutral is None:
        raise ValueError("At least one representativeness constraint must be provided.")
    if min_neff_fraction is not None and min_neff_fraction <= 0.0:
        raise ValueError(f"min_neff_fraction must be positive, got {min_neff_fraction!r}.")
    if max_kl_to_neutral is not None and max_kl_to_neutral < 0.0:
        raise ValueError(f"max_kl_to_neutral must be nonnegative, got {max_kl_to_neutral!r}.")
    if grid_size < 2:
        raise ValueError(f"grid_size must be at least 2, got {grid_size!r}.")

    weights, z_hat, z_ref = _compute_qvfit_row_terms(
        support_keys,
        betas,
        ref_keys,
        query_bank,
    )
    for step in range(grid_size):
        gamma = 1.0 - (step / (grid_size - 1))
        stats = _compute_qvfit_row_scaling_stats_from_terms(
            weights,
            z_hat,
            z_ref,
            row_scale_power=gamma,
        )
        ok = True
        if min_neff_fraction is not None:
            ok = ok and (stats["q_weight_neff_fraction"] >= min_neff_fraction)
        if max_kl_to_neutral is not None:
            ok = ok and (stats["q_weight_kl_to_neutral"] <= max_kl_to_neutral)
        if ok:
            return gamma, stats
    return 0.0, _compute_qvfit_row_scaling_stats_from_terms(
        weights,
        z_hat,
        z_ref,
        row_scale_power=0.0,
    )


def choose_diagnostic_qfit_row_scale_power(
    support_keys: Tensor,
    betas: Tensor,
    ref_keys: Tensor,
    query_bank: QueryBank,
    *,
    full_metric_max_kl_to_neutral: float,
    hard_gate_zhat_over_zref_cv: float,
    middle_control: str = "kl",
    middle_min_neff_fraction: float = 0.5,
    middle_max_kl_to_neutral: float = 0.25,
    grid_size: int = 65,
) -> tuple[float, dict[str, float], str]:
    """Choose a diagnostic-conditioned row metric for qfit.

    Policy:
    - if the full quotient metric is already close enough to the neutral bank
      metric, use full qvfit
    - if quotient row-scaling dispersion is clearly too large, fall back to the
      neutral metric
    - otherwise use the strongest admissible smooth middle control
    """
    if middle_control not in {"kl", "neff"}:
        raise ValueError(f"Unsupported middle_control: {middle_control!r}.")

    full_stats = compute_qvfit_row_scaling_stats(
        support_keys,
        betas,
        ref_keys,
        query_bank,
        row_scale_power=1.0,
    )
    if full_stats["q_weight_kl_to_neutral"] <= full_metric_max_kl_to_neutral:
        return 1.0, full_stats, "full_quotient"

    if full_stats["zhat_over_zref_cv"] >= hard_gate_zhat_over_zref_cv:
        neutral_stats = compute_qvfit_row_scaling_stats(
            support_keys,
            betas,
            ref_keys,
            query_bank,
            row_scale_power=0.0,
        )
        return 0.0, neutral_stats, "neutral_fallback"

    if middle_control == "kl":
        gamma, stats = choose_qvfit_row_scale_power(
            support_keys,
            betas,
            ref_keys,
            query_bank,
            max_kl_to_neutral=middle_max_kl_to_neutral,
            grid_size=grid_size,
        )
    else:
        gamma, stats = choose_qvfit_row_scale_power(
            support_keys,
            betas,
            ref_keys,
            query_bank,
            min_neff_fraction=middle_min_neff_fraction,
            grid_size=grid_size,
        )
    branch = "neutral_fallback" if gamma == 0.0 else f"middle_{middle_control}"
    return gamma, stats, branch


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
