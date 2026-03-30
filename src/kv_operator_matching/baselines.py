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
from dataclasses import dataclass

import torch

from .objectives import compute_logits
from .query_bank import QueryBank
from .types import CompactRepresentation, HeadState


@dataclass(frozen=True)
class HybridSelectorConfig:
    """Configuration for the Phase 3A hybrid support selector."""

    use_delta_b: bool = True
    use_delta_q_coh: bool = True
    use_delta_q_span: bool = False
    use_delta_q_low_sv_risk: bool = False
    use_delta_q_redundancy: bool = False
    use_evidence_weights: bool = True
    fixed_alpha: float | None = None
    fixed_beta: float | None = None


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


def hybrid_support_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    config: HybridSelectorConfig | None = None,
) -> CompactRepresentation:
    """Continuous hybrid selector over original-token candidates only.

    The live Phase 3A selector uses a greedy additive score centered on the
    current core terms:

        J_add = Delta B + alpha(E) * Delta Q_coh

    Optional penalties such as span or other risk proxies are still exposed
    through HybridSelectorConfig for ablation and future follow-up work.

    `alpha(E)` and `beta(E)` are continuous functions of evidence-state
    observables derived from the current query bank.
    """
    if config is None:
        config = HybridSelectorConfig()

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

    weighted_design, weighted_target = _build_mass_frame(queries_f, keys_f, weights_f)
    alpha, beta = hybrid_evidence_weights(
        queries_f,
        weights_f,
        use_evidence_weights=config.use_evidence_weights,
        fixed_alpha=config.fixed_alpha,
        fixed_beta=config.fixed_beta,
    )

    selected_indices: list[int] = []
    selected_mask = torch.zeros(n, dtype=torch.bool, device=keys_f.device)
    current_prediction = torch.zeros_like(weighted_target)
    current_span_frac = 0.0
    current_min = None
    current_max = None

    column_norm_sq = weighted_design.pow(2).sum(dim=0).clamp_min(1e-12)
    all_indices = torch.arange(n, device=keys_f.device)
    normalized_keys = torch.nn.functional.normalize(keys_f, dim=1)

    for _ in range(m):
        residual = weighted_target - current_prediction
        residual_energy = residual.pow(2).sum().clamp_min(1e-12)
        corr = torch.matmul(weighted_design.T, residual)
        delta_b = corr.clamp_min(0.0).pow(2) / (column_norm_sq * residual_energy)

        if selected_indices:
            selected_design = weighted_design[:, selected_indices]
            q_basis = torch.linalg.qr(selected_design, mode="reduced").Q
            projected = q_basis.T @ weighted_design
            projected_norm_sq = projected.pow(2).sum(dim=0)
            orth_norm_sq = (column_norm_sq - projected_norm_sq).clamp_min(0.0)
            delta_q_coh = orth_norm_sq / column_norm_sq
        else:
            delta_q_coh = torch.ones_like(delta_b)

        if config.use_delta_q_low_sv_risk and selected_indices:
            left_singular, _s, _vh = torch.linalg.svd(selected_design, full_matrices=False)
            coeff = left_singular.T @ weighted_design
            projected_energy = coeff.pow(2).sum(dim=0)
            cutoff = max(1, coeff.shape[0] // 4)
            low_energy = coeff[-cutoff:].pow(2).sum(dim=0)
            delta_q_low_sv_risk = torch.where(
                projected_energy > 1e-12,
                low_energy / projected_energy.clamp_min(1e-12),
                torch.zeros_like(projected_energy),
            )
        else:
            delta_q_low_sv_risk = torch.zeros_like(delta_b)

        if config.use_delta_q_redundancy and selected_indices:
            selected_key_basis = normalized_keys[selected_indices]
            key_similarity = normalized_keys @ selected_key_basis.T
            delta_q_redundancy = key_similarity.clamp_min(0.0).max(dim=1).values
        else:
            delta_q_redundancy = torch.zeros_like(delta_b)

        if current_min is None or current_max is None:
            new_span_frac = torch.full_like(delta_b, 1.0 / max(n, 1))
        else:
            new_min = torch.minimum(all_indices, torch.full_like(all_indices, current_min))
            new_max = torch.maximum(all_indices, torch.full_like(all_indices, current_max))
            new_span_frac = (new_max - new_min + 1).float() / max(n, 1)
        delta_q_span = (new_span_frac - current_span_frac).clamp_min(0.0)

        score = torch.zeros_like(delta_b)
        if config.use_delta_b:
            score = score + delta_b
        if config.use_delta_q_coh:
            score = score + alpha * delta_q_coh
        if config.use_delta_q_span:
            score = score - beta * delta_q_span
        if config.use_delta_q_low_sv_risk:
            score = score - beta * delta_q_low_sv_risk
        if config.use_delta_q_redundancy:
            score = score - beta * delta_q_redundancy
        score[selected_mask] = -float("inf")
        index = int(torch.argmax(score).item())
        if not math.isfinite(float(score[index].item())):
            break

        selected_indices.append(index)
        selected_mask[index] = True
        current_min = index if current_min is None else min(current_min, index)
        current_max = index if current_max is None else max(current_max, index)
        current_span_frac = float(((current_max - current_min + 1) / max(n, 1)))

        selected_design = weighted_design[:, selected_indices]
        scale = torch.linalg.lstsq(
            selected_design,
            weighted_target.unsqueeze(1),
            driver="gels",
        ).solution.squeeze(1)
        scale = scale.clamp_min(1e-12)
        current_prediction = selected_design @ scale

    if not selected_indices:
        return recency_baseline(head_state, budget)

    index_tensor = torch.tensor(selected_indices, device=keys.device, dtype=torch.long)
    support_keys = keys[index_tensor]
    support_values = values[index_tensor]
    selected_design = weighted_design[:, selected_indices]
    betas = torch.linalg.lstsq(
        selected_design,
        weighted_target.unsqueeze(1),
        driver="gels",
    ).solution.squeeze(1).clamp_min(1e-12).to(dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def hybrid_pairmerge_support_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    config: HybridSelectorConfig | None = None,
) -> CompactRepresentation:
    """Hybrid selector over source-grounded original and adjacent-pair atoms.

    Candidate pool:
    - every original token as a singleton atom
    - every adjacent pair `(i, i+1)` merged into one atom using simple means

    Selection is conflict-aware: once an atom covering source positions
    `[start, end]` is selected, no other candidate overlapping that span may be
    selected. This keeps the support source-grounded and avoids double-counting
    the same original KV entries through both singleton and merged atoms.
    """
    if config is None:
        config = HybridSelectorConfig()

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
    values_f = values.float()
    weights_f = weights.float()

    singleton_keys = keys_f
    singleton_values = values_f
    singleton_starts = torch.arange(n, device=keys_f.device, dtype=torch.long)
    singleton_ends = singleton_starts.clone()

    if n > 1:
        pair_keys = 0.5 * (keys_f[:-1] + keys_f[1:])
        pair_values = 0.5 * (values_f[:-1] + values_f[1:])
        pair_starts = torch.arange(n - 1, device=keys_f.device, dtype=torch.long)
        pair_ends = pair_starts + 1

        candidate_keys = torch.cat([singleton_keys, pair_keys], dim=0)
        candidate_values = torch.cat([singleton_values, pair_values], dim=0)
        candidate_starts = torch.cat([singleton_starts, pair_starts], dim=0)
        candidate_ends = torch.cat([singleton_ends, pair_ends], dim=0)
    else:
        candidate_keys = singleton_keys
        candidate_values = singleton_values
        candidate_starts = singleton_starts
        candidate_ends = singleton_ends

    weighted_design, weighted_target = _build_candidate_mass_frame(
        query_tensor=queries_f,
        target_key_tensor=keys_f,
        candidate_key_tensor=candidate_keys,
        entry_weights=weights_f,
    )
    alpha, beta = hybrid_evidence_weights(
        queries_f,
        weights_f,
        use_evidence_weights=config.use_evidence_weights,
        fixed_alpha=config.fixed_alpha,
        fixed_beta=config.fixed_beta,
    )

    selected_indices: list[int] = []
    selected_mask = torch.zeros(candidate_keys.shape[0], dtype=torch.bool, device=keys_f.device)
    occupied = torch.zeros(n, dtype=torch.bool, device=keys_f.device)
    current_prediction = torch.zeros_like(weighted_target)
    current_span_frac = 0.0
    current_min = None
    current_max = None

    column_norm_sq = weighted_design.pow(2).sum(dim=0).clamp_min(1e-12)
    normalized_candidate_keys = torch.nn.functional.normalize(candidate_keys, dim=1)

    for _ in range(m):
        residual = weighted_target - current_prediction
        residual_energy = residual.pow(2).sum().clamp_min(1e-12)
        corr = torch.matmul(weighted_design.T, residual)
        delta_b = corr.clamp_min(0.0).pow(2) / (column_norm_sq * residual_energy)

        if selected_indices:
            selected_design = weighted_design[:, selected_indices]
            q_basis = torch.linalg.qr(selected_design, mode="reduced").Q
            projected = q_basis.T @ weighted_design
            projected_norm_sq = projected.pow(2).sum(dim=0)
            orth_norm_sq = (column_norm_sq - projected_norm_sq).clamp_min(0.0)
            delta_q_coh = orth_norm_sq / column_norm_sq
        else:
            delta_q_coh = torch.ones_like(delta_b)

        if config.use_delta_q_low_sv_risk and selected_indices:
            left_singular, _s, _vh = torch.linalg.svd(selected_design, full_matrices=False)
            coeff = left_singular.T @ weighted_design
            projected_energy = coeff.pow(2).sum(dim=0)
            cutoff = max(1, coeff.shape[0] // 4)
            low_energy = coeff[-cutoff:].pow(2).sum(dim=0)
            delta_q_low_sv_risk = torch.where(
                projected_energy > 1e-12,
                low_energy / projected_energy.clamp_min(1e-12),
                torch.zeros_like(projected_energy),
            )
        else:
            delta_q_low_sv_risk = torch.zeros_like(delta_b)

        if config.use_delta_q_redundancy and selected_indices:
            selected_key_basis = normalized_candidate_keys[selected_indices]
            key_similarity = normalized_candidate_keys @ selected_key_basis.T
            delta_q_redundancy = key_similarity.clamp_min(0.0).max(dim=1).values
        else:
            delta_q_redundancy = torch.zeros_like(delta_b)

        if current_min is None or current_max is None:
            new_span_frac = (candidate_ends - candidate_starts + 1).float() / max(n, 1)
        else:
            new_min = torch.minimum(candidate_starts, torch.full_like(candidate_starts, current_min))
            new_max = torch.maximum(candidate_ends, torch.full_like(candidate_ends, current_max))
            new_span_frac = (new_max - new_min + 1).float() / max(n, 1)
        delta_q_span = (new_span_frac - current_span_frac).clamp_min(0.0)

        occupied_prefix = torch.cat(
            [
                torch.zeros(1, device=occupied.device, dtype=torch.long),
                occupied.to(dtype=torch.long).cumsum(dim=0),
            ]
        )
        overlap = (occupied_prefix[candidate_ends + 1] - occupied_prefix[candidate_starts]) > 0

        score = torch.zeros_like(delta_b)
        if config.use_delta_b:
            score = score + delta_b
        if config.use_delta_q_coh:
            score = score + alpha * delta_q_coh
        if config.use_delta_q_span:
            score = score - beta * delta_q_span
        if config.use_delta_q_low_sv_risk:
            score = score - beta * delta_q_low_sv_risk
        if config.use_delta_q_redundancy:
            score = score - beta * delta_q_redundancy
        score[selected_mask | overlap] = -float("inf")

        index = int(torch.argmax(score).item())
        if not math.isfinite(float(score[index].item())):
            break

        selected_indices.append(index)
        selected_mask[index] = True
        start = int(candidate_starts[index].item())
        end = int(candidate_ends[index].item())
        occupied[start : end + 1] = True
        current_min = start if current_min is None else min(current_min, start)
        current_max = end if current_max is None else max(current_max, end)
        current_span_frac = float(((current_max - current_min + 1) / max(n, 1)))

        selected_design = weighted_design[:, selected_indices]
        scale = torch.linalg.lstsq(
            selected_design,
            weighted_target.unsqueeze(1),
            driver="gels",
        ).solution.squeeze(1)
        scale = scale.clamp_min(1e-12)
        current_prediction = selected_design @ scale

    if not selected_indices:
        return hybrid_support_baseline(head_state, query_bank, budget, config)

    index_tensor = torch.tensor(selected_indices, device=keys.device, dtype=torch.long)
    support_keys = candidate_keys[index_tensor].to(dtype=keys.dtype, device=keys.device)
    support_values = candidate_values[index_tensor].to(dtype=values.dtype, device=values.device)
    selected_design = weighted_design[:, selected_indices]
    betas = torch.linalg.lstsq(
        selected_design,
        weighted_target.unsqueeze(1),
        driver="gels",
    ).solution.squeeze(1).clamp_min(1e-12).to(dtype=keys.dtype, device=keys.device)
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


def _build_mass_frame(
    query_tensor: torch.Tensor,
    key_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    logits = (query_tensor @ key_tensor.T) * inv_sqrt_d
    reference_max = logits.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(logits - reference_max)
    target = exp_scores.sum(dim=1)
    row_weights = torch.sqrt(torch.clamp_min(entry_weights.to(dtype=torch.float32), 0.0))
    weighted_design = exp_scores * row_weights.unsqueeze(1)
    weighted_target = target * row_weights
    return weighted_design, weighted_target


def _build_candidate_mass_frame(
    query_tensor: torch.Tensor,
    target_key_tensor: torch.Tensor,
    candidate_key_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the stable exp-logit mass frame for a constructed candidate pool."""
    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    target_logits = (query_tensor @ target_key_tensor.T) * inv_sqrt_d
    reference_max = target_logits.max(dim=1, keepdim=True).values
    target = torch.exp(target_logits - reference_max).sum(dim=1)
    candidate_logits = (query_tensor @ candidate_key_tensor.T) * inv_sqrt_d
    candidate_exp_scores = torch.exp(candidate_logits - reference_max)
    row_weights = torch.sqrt(torch.clamp_min(entry_weights.to(dtype=torch.float32), 0.0))
    weighted_design = candidate_exp_scores * row_weights.unsqueeze(1)
    weighted_target = target * row_weights
    return weighted_design, weighted_target


def hybrid_evidence_weights(
    query_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
    *,
    use_evidence_weights: bool = True,
    fixed_alpha: float | None = None,
    fixed_beta: float | None = None,
) -> tuple[float, float]:
    """Map continuous query-bank observables to coherence/span weights."""
    if fixed_alpha is not None or fixed_beta is not None:
        alpha = 0.0 if fixed_alpha is None else float(fixed_alpha)
        beta = 0.0 if fixed_beta is None else float(fixed_beta)
        return alpha, beta
    if not use_evidence_weights:
        return 1.0, 1.0

    weighted_queries = query_tensor * entry_weights.sqrt().unsqueeze(-1)
    sv = torch.linalg.svdvals(weighted_queries)
    smax = float(sv.max().item()) if sv.numel() > 0 else 0.0
    srank = float((weighted_queries.square().sum() / max(smax * smax, 1e-12)).item())
    srank_norm = min(
        srank / max(1.0, float(min(weighted_queries.shape[0], weighted_queries.shape[1]))),
        1.0,
    )
    weight_sum = float(entry_weights.sum().item())
    ess = (weight_sum * weight_sum) / max(float(entry_weights.square().sum().item()), 1e-12)
    ess_norm = min(ess / max(float(query_tensor.shape[0]), 1.0), 1.0)
    richness = 0.5 * (srank_norm + ess_norm)
    alpha = 0.2 + 0.8 * (1.0 - richness)
    beta = 0.2 + 0.8 * (1.0 - richness)
    return alpha, beta


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
