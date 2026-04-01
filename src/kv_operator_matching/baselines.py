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
    keys = head_state.keys
    values = head_state.values
    n = keys.shape[0]
    m = min(budget, n)
    mass = compute_attention_mass_scores(head_state, query_bank)
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


def compute_attention_mass_scores(
    head_state: HeadState,
    query_bank: QueryBank,
) -> torch.Tensor:
    """Return aggregated attention-mass scores over the query bank."""
    queries, weights = query_bank.get_weighted_bank()
    queries_f = queries.float()
    keys_f = head_state.keys.float()
    weights_f = weights.float()
    logits = compute_logits(queries_f, keys_f)
    attn_weights = torch.softmax(logits, dim=-1)
    return (weights_f.unsqueeze(-1) * attn_weights).sum(dim=0)


def compute_value_deviation_scores(
    head_state: HeadState,
    query_bank: QueryBank,
) -> torch.Tensor:
    """Return mean value-deviation scores conditioned on received attention."""
    queries, weights = query_bank.get_weighted_bank()
    queries_f = queries.float()
    keys_f = head_state.keys.float()
    values_f = head_state.values.float()
    weights_f = weights.float()
    logits = compute_logits(queries_f, keys_f)
    attn_weights = torch.softmax(logits, dim=-1)
    outputs = attn_weights @ values_f
    value_gap_sq = (values_f.unsqueeze(0) - outputs.unsqueeze(1)).pow(2).sum(dim=-1)
    numer = (weights_f.unsqueeze(-1) * attn_weights * value_gap_sq).sum(dim=0)
    denom = (weights_f.unsqueeze(-1) * attn_weights).sum(dim=0).clamp_min(1e-12)
    return numer / denom


def compute_quotient_omission_scores(
    head_state: HeadState,
    query_bank: QueryBank,
    *,
    exact_local: bool = True,
    clamp_eps: float = 1e-6,
) -> torch.Tensor:
    """Return bank-aggregated quotient-aware omission scores."""
    queries, weights = query_bank.get_weighted_bank()
    queries_f = queries.float()
    keys_f = head_state.keys.float()
    values_f = head_state.values.float()
    weights_f = weights.float()

    logits = compute_logits(queries_f, keys_f)
    attn_weights = torch.softmax(logits, dim=-1)
    outputs = attn_weights @ values_f
    value_gap_sq = (values_f.unsqueeze(0) - outputs.unsqueeze(1)).pow(2).sum(dim=-1)
    if exact_local:
        omission_scale = attn_weights / (1.0 - attn_weights).clamp(min=clamp_eps)
        score_terms = omission_scale.square() * value_gap_sq
    else:
        score_terms = attn_weights * value_gap_sq
    return (weights_f.unsqueeze(-1) * score_terms).sum(dim=0)


def shortlist_indices_from_scores(
    attention_scores: torch.Tensor,
    quotient_scores: torch.Tensor,
    shortlist_size: int,
    *,
    policy: str,
    gate_expansion: int = 2,
) -> torch.Tensor:
    """Build shortlist indices for a structural shortlist policy."""
    n = attention_scores.shape[0]
    shortlist_size = min(max(int(shortlist_size), 1), n)
    if policy == "attn_mass":
        return torch.topk(attention_scores, k=shortlist_size).indices
    if policy == "quotient_omit":
        return torch.topk(quotient_scores, k=shortlist_size).indices
    if policy == "rank_blend":
        attn_order = torch.argsort(attention_scores, descending=True)
        quot_order = torch.argsort(quotient_scores, descending=True)
        attn_rank = torch.empty_like(attn_order)
        quot_rank = torch.empty_like(quot_order)
        attn_rank[attn_order] = torch.arange(n, device=attn_order.device, dtype=attn_order.dtype)
        quot_rank[quot_order] = torch.arange(n, device=quot_order.device, dtype=quot_order.dtype)
        rank_sum = attn_rank.to(dtype=torch.float32) + quot_rank.to(dtype=torch.float32)
        return torch.topk(-rank_sum, k=shortlist_size).indices
    if policy == "two_stage_gate":
        gate_size = min(n, max(shortlist_size, int(gate_expansion * shortlist_size)))
        gate = torch.topk(attention_scores, k=gate_size).indices
        gated_scores = quotient_scores[gate]
        gated_keep = torch.topk(gated_scores, k=shortlist_size).indices
        return gate[gated_keep]
    raise ValueError(f"Unsupported shortlist policy: {policy}")


def omp_over_shortlist(
    head_state: HeadState,
    query_bank: QueryBank,
    shortlist_indices: torch.Tensor,
    budget: int,
) -> CompactRepresentation:
    """Run the existing mass-frame OMP inside a fixed candidate shortlist."""
    queries, weights = query_bank.get_weighted_bank()
    keys = head_state.keys
    values = head_state.values
    shortlist_indices = shortlist_indices.to(device=keys.device, dtype=torch.long)
    shortlist_keys_f = keys[shortlist_indices].float()
    selected_local, selected_betas = _select_keys_with_omp(
        key_tensor=shortlist_keys_f,
        query_tensor=queries.float(),
        entry_weights=weights.float(),
        selection_budget=min(budget, shortlist_indices.numel()),
    )
    if not selected_local:
        return attention_mass_baseline(head_state, query_bank, budget)
    local_index_tensor = torch.tensor(selected_local, device=shortlist_indices.device, dtype=torch.long)
    index_tensor = shortlist_indices[local_index_tensor]
    support_keys = keys[index_tensor]
    support_values = values[index_tensor]
    betas = torch.tensor(selected_betas, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def quotient_omission_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    *,
    exact_local: bool = True,
    clamp_eps: float = 1e-6,
) -> CompactRepresentation:
    """Select support by bank-aggregated quotient-aware omission damage.

    This baseline keeps the tokens whose removal would most damage the local
    attention output over the query bank. For each query q and token i, it
    scores the exact local single-atom omission error:

        ||O_{-i}(q) - O(q)||^2
          = (alpha_i(q) / (1 - alpha_i(q)))^2 * ||v_i - O(q)||^2

    where alpha_i(q) is the full-cache normalized attention weight and O(q) is
    the full attention output. This is the local normalized counterpart to the
    quotient-residual omission identity E_i(q) = -w_i(q) (v_i - O(q)).

    If exact_local is False, the score falls back to the cheaper proxy

        alpha_i(q) * ||v_i - O(q)||^2

    which still captures the missing value-sensitive factor ignored by the
    attention-mass baseline.
    """
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

    score = compute_quotient_omission_scores(
        head_state,
        query_bank,
        exact_local=exact_local,
        clamp_eps=clamp_eps,
    )
    top_indices = torch.topk(score, k=m).indices

    support_keys = keys[top_indices]
    support_values = values[top_indices]
    betas = torch.ones(m, dtype=keys.dtype, device=keys.device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def quotient_omit_omp_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    *,
    shortlist_multiplier: float = 2.0,
    exact_local: bool = True,
    clamp_eps: float = 1e-6,
) -> CompactRepresentation:
    """Run OMP inside a quotient-aware shortlist.

    This is the conservative selector sequencing suggested by the
    quotient-residual note:

    1. rank original tokens by bank-aggregated quotient-aware omission damage
    2. keep only the top-k candidates for a small shortlist
    3. run the existing mass-frame OMP inside that shortlist

    The goal is to test whether value-aware pre-screening is already enough to
    improve support quality before introducing a full E-aware OMP objective.
    """
    keys = head_state.keys
    n = keys.shape[0]
    m = min(budget, n)
    if m <= 0:
        return CompactRepresentation(
            support_keys=keys[:0],
            support_values=head_state.values[:0],
            betas=torch.ones(0, dtype=keys.dtype, device=keys.device),
        )

    shortlist_size = min(n, max(m, int(round(shortlist_multiplier * m))))
    shortlist_indices = shortlist_indices_from_scores(
        compute_attention_mass_scores(head_state, query_bank),
        compute_quotient_omission_scores(
            head_state,
            query_bank,
            exact_local=exact_local,
            clamp_eps=clamp_eps,
        ),
        shortlist_size,
        policy="quotient_omit",
    )
    return omp_over_shortlist(
        head_state,
        query_bank,
        shortlist_indices,
        m,
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

    return _hybrid_support_over_candidate_pool(
        target_keys=keys,
        query_tensor=queries_f,
        entry_weights=weights_f,
        candidate_keys=candidate_keys,
        candidate_values=candidate_values,
        candidate_starts=candidate_starts,
        candidate_ends=candidate_ends,
        budget=m,
        config=config,
        fallback_rep=hybrid_support_baseline(head_state, query_bank, budget, config),
    )


def hybrid_fitted_pairmerge_support_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    config: HybridSelectorConfig | None = None,
    *,
    mass_cos_threshold: float = 0.95,
    value_cos_threshold: float = 0.75,
) -> CompactRepresentation:
    """Hybrid selector over original tokens plus fitted adjacent-pair atoms."""
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

    eligible_pairs, _mass_cos, _value_cos = compute_adjacent_pair_compatibility(
        head_state,
        query_bank,
        mass_cos_threshold=mass_cos_threshold,
        value_cos_threshold=value_cos_threshold,
    )
    eligible_indices = torch.nonzero(eligible_pairs, as_tuple=False).squeeze(1)

    if eligible_indices.numel() > 0:
        fitted_pair_keys = []
        fitted_pair_values = []
        for pair_idx in eligible_indices.tolist():
            pair_key, pair_value = _fit_adjacent_pair_representative(
                query_tensor=queries_f,
                entry_weights=weights_f,
                left_key=keys_f[pair_idx],
                right_key=keys_f[pair_idx + 1],
                left_value=values_f[pair_idx],
                right_value=values_f[pair_idx + 1],
            )
            fitted_pair_keys.append(pair_key.unsqueeze(0))
            fitted_pair_values.append(pair_value.unsqueeze(0))

        pair_keys = torch.cat(fitted_pair_keys, dim=0)
        pair_values = torch.cat(fitted_pair_values, dim=0)
        pair_starts = eligible_indices.to(dtype=torch.long, device=keys_f.device)
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

    return _hybrid_support_over_candidate_pool(
        target_keys=keys,
        query_tensor=queries_f,
        entry_weights=weights_f,
        candidate_keys=candidate_keys,
        candidate_values=candidate_values,
        candidate_starts=candidate_starts,
        candidate_ends=candidate_ends,
        budget=m,
        config=config,
        fallback_rep=hybrid_support_baseline(head_state, query_bank, budget, config),
    )


def hybrid_anchor_region_support_baseline(
    head_state: HeadState,
    query_bank: QueryBank,
    budget: int,
    config: HybridSelectorConfig | None = None,
    *,
    assignment_window: int = 8,
    assignment_distance_penalty: float = 0.15,
    assignment_score_floor: float = 0.0,
    max_neighbor_blend: float = 0.5,
) -> CompactRepresentation:
    """Construct one conservative regional representative around each hybrid anchor.

    This keeps the Phase 3A selector fixed. It first selects the live original-
    token hybrid support, then replaces each selected anchor with a softly
    blended regional representative built from nearby assigned tokens.

    The intent is to test anchor-conditioned local construction, not a richer
    selector. Assignment remains local and low-capacity by design.
    """
    if config is None:
        config = HybridSelectorConfig()

    base_rep = hybrid_support_baseline(head_state, query_bank, budget, config)
    if base_rep.support_keys.numel() == 0:
        return base_rep

    queries, weights = query_bank.get_weighted_bank()
    keys = head_state.keys
    values = head_state.values
    queries_f = queries.float()
    keys_f = keys.float()
    values_f = values.float()
    weights_f = weights.float()

    weighted_design, weighted_target = _build_mass_frame(queries_f, keys_f, weights_f)
    anchor_indices = _match_support_to_source_indices(keys_f, base_rep.support_keys.float())
    region = _build_anchor_region_atoms(
        keys_f=keys_f,
        values_f=values_f,
        weighted_design=weighted_design,
        anchor_indices=anchor_indices,
        assignment_window=assignment_window,
        assignment_distance_penalty=assignment_distance_penalty,
        assignment_score_floor=assignment_score_floor,
        max_neighbor_blend=max_neighbor_blend,
    )

    candidate_design, candidate_target = _build_candidate_mass_frame(
        query_tensor=queries_f,
        target_key_tensor=keys_f,
        candidate_key_tensor=region["support_keys"],
        entry_weights=weights_f,
    )
    betas = torch.linalg.lstsq(
        candidate_design,
        candidate_target.unsqueeze(1),
        driver="gels",
    ).solution.squeeze(1).clamp_min(1e-12)
    return CompactRepresentation(
        support_keys=region["support_keys"].to(dtype=keys.dtype, device=keys.device),
        support_values=region["support_values"].to(dtype=values.dtype, device=values.device),
        betas=betas.to(dtype=keys.dtype, device=keys.device),
    )


def compute_adjacent_pair_compatibility(
    head_state: HeadState,
    query_bank: QueryBank,
    *,
    mass_cos_threshold: float = 0.95,
    value_cos_threshold: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return eligibility and component similarities for adjacent pair merges."""
    queries, weights = query_bank.get_weighted_bank()
    keys_f = head_state.keys.float()
    values_f = head_state.values.float()
    queries_f = queries.float()
    weights_f = weights.float()
    n = keys_f.shape[0]
    if n <= 1:
        empty = torch.empty(0, dtype=torch.float32, device=keys_f.device)
        return torch.zeros(0, dtype=torch.bool, device=keys_f.device), empty, empty

    weighted_design, _weighted_target = _build_mass_frame(queries_f, keys_f, weights_f)
    left_cols = weighted_design[:, :-1]
    right_cols = weighted_design[:, 1:]
    left_norm = left_cols.norm(dim=0).clamp_min(1e-12)
    right_norm = right_cols.norm(dim=0).clamp_min(1e-12)
    mass_cos = (left_cols * right_cols).sum(dim=0) / (left_norm * right_norm)

    normalized_values = torch.nn.functional.normalize(values_f, dim=1)
    value_cos = (normalized_values[:-1] * normalized_values[1:]).sum(dim=1)

    eligible = (mass_cos >= mass_cos_threshold) & (value_cos >= value_cos_threshold)
    return eligible, mass_cos, value_cos


def _match_support_to_source_indices(
    source_keys: torch.Tensor,
    support_keys: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> list[int]:
    """Map support keys drawn from the source cache back to source indices."""
    if support_keys.numel() == 0:
        return []

    remaining = torch.ones(source_keys.shape[0], dtype=torch.bool, device=source_keys.device)
    matched: list[int] = []
    for support_key in support_keys:
        diff = (source_keys - support_key.unsqueeze(0)).abs().amax(dim=1)
        if remaining.any():
            masked = diff.masked_fill(~remaining, float("inf"))
            index = int(torch.argmin(masked).item())
            if math.isfinite(float(masked[index].item())):
                matched.append(index)
                remaining[index] = False
                continue
        index = int(torch.argmin(diff).item())
        matched.append(index)
        remaining[index] = False
    return matched


def _build_anchor_region_atoms(
    *,
    keys_f: torch.Tensor,
    values_f: torch.Tensor,
    weighted_design: torch.Tensor,
    anchor_indices: list[int],
    assignment_window: int,
    assignment_distance_penalty: float,
    assignment_score_floor: float,
    max_neighbor_blend: float,
) -> dict[str, torch.Tensor]:
    """Construct one soft local representative around each selected anchor."""
    if not anchor_indices:
        return {
            "support_keys": keys_f[:0],
            "support_values": values_f[:0],
            "assignments": torch.empty(0, dtype=torch.long, device=keys_f.device),
            "assignment_scores": torch.empty(0, dtype=torch.float32, device=keys_f.device),
            "region_sizes": torch.empty(0, dtype=torch.long, device=keys_f.device),
        }

    device = keys_f.device
    dtype = keys_f.dtype
    anchor_index_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
    anchor_cols = weighted_design[:, anchor_index_t]
    column_norm = weighted_design.norm(dim=0).clamp_min(1e-12)
    anchor_norm = anchor_cols.norm(dim=0).clamp_min(1e-12)
    role_cos = (weighted_design.T @ anchor_cols) / (column_norm.unsqueeze(1) * anchor_norm.unsqueeze(0))

    token_indices = torch.arange(keys_f.shape[0], device=device, dtype=torch.long)
    distances = (token_indices.unsqueeze(1) - anchor_index_t.unsqueeze(0)).abs()
    if assignment_window <= 0:
        local_mask = distances == 0
        distance_penalty = torch.zeros_like(role_cos)
    else:
        local_mask = distances <= assignment_window
        denom = max(assignment_window - 1, 1)
        distance_penalty = assignment_distance_penalty * (
            distances.clamp_min(1).float() - 1.0
        ) / denom

    score = role_cos - distance_penalty
    score = score.masked_fill(~local_mask, -float("inf"))
    for anchor_pos, anchor_idx in enumerate(anchor_indices):
        score[anchor_idx, anchor_pos] = float("inf")

    best_score, best_anchor_pos = torch.max(score, dim=1)
    assigned_mask = best_score > assignment_score_floor
    assigned_mask[anchor_index_t] = True
    assignments = torch.full((keys_f.shape[0],), -1, dtype=torch.long, device=device)
    assignments[assigned_mask] = best_anchor_pos[assigned_mask]

    support_keys = []
    support_values = []
    region_sizes = []
    assignment_scores = torch.zeros(keys_f.shape[0], dtype=torch.float32, device=device)
    assignment_scores[assigned_mask] = torch.where(
        torch.isfinite(best_score[assigned_mask]),
        best_score[assigned_mask].to(dtype=torch.float32),
        torch.ones_like(best_score[assigned_mask], dtype=torch.float32),
    )

    for anchor_pos, anchor_idx in enumerate(anchor_indices):
        member_mask = assignments == anchor_pos
        members = torch.nonzero(member_mask, as_tuple=False).squeeze(1)
        if members.numel() == 0:
            members = torch.tensor([anchor_idx], dtype=torch.long, device=device)
        region_sizes.append(int(members.numel()))

        member_scores = assignment_scores[members].clamp_min(0.0)
        anchor_member = members == anchor_idx
        if anchor_member.any():
            member_scores[anchor_member] = 1.0
        else:
            members = torch.cat(
                [torch.tensor([anchor_idx], dtype=torch.long, device=device), members], dim=0
            )
            member_scores = torch.cat(
                [torch.ones(1, dtype=torch.float32, device=device), member_scores], dim=0
            )

        weight_sum = member_scores.sum().clamp_min(1e-12)
        mean_key = (member_scores.unsqueeze(1) * keys_f[members]).sum(dim=0) / weight_sum
        mean_value = (member_scores.unsqueeze(1) * values_f[members]).sum(dim=0) / weight_sum

        neighbor_weight = member_scores[(members != anchor_idx)].sum()
        blend = float(
            min(
                max_neighbor_blend,
                float((neighbor_weight / (1.0 + neighbor_weight)).item()) if neighbor_weight.numel() > 0 else 0.0,
            )
        )
        anchor_key = keys_f[anchor_idx]
        anchor_value = values_f[anchor_idx]
        support_keys.append(((1.0 - blend) * anchor_key + blend * mean_key).unsqueeze(0))
        support_values.append(((1.0 - blend) * anchor_value + blend * mean_value).unsqueeze(0))

    return {
        "support_keys": torch.cat(support_keys, dim=0).to(dtype=dtype, device=device),
        "support_values": torch.cat(support_values, dim=0).to(dtype=values_f.dtype, device=values_f.device),
        "assignments": assignments,
        "assignment_scores": assignment_scores,
        "region_sizes": torch.tensor(region_sizes, dtype=torch.long, device=device),
    }


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


def _hybrid_support_over_candidate_pool(
    *,
    target_keys: torch.Tensor,
    query_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
    candidate_keys: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_starts: torch.Tensor,
    candidate_ends: torch.Tensor,
    budget: int,
    config: HybridSelectorConfig,
    fallback_rep: CompactRepresentation,
) -> CompactRepresentation:
    weighted_design, weighted_target = _build_candidate_mass_frame(
        query_tensor=query_tensor,
        target_key_tensor=target_keys.float(),
        candidate_key_tensor=candidate_keys,
        entry_weights=entry_weights,
    )
    alpha, beta = hybrid_evidence_weights(
        query_tensor,
        entry_weights,
        use_evidence_weights=config.use_evidence_weights,
        fixed_alpha=config.fixed_alpha,
        fixed_beta=config.fixed_beta,
    )

    n_tokens = int(target_keys.shape[0])
    selected_indices: list[int] = []
    selected_mask = torch.zeros(candidate_keys.shape[0], dtype=torch.bool, device=candidate_keys.device)
    occupied = torch.zeros(n_tokens, dtype=torch.bool, device=candidate_keys.device)
    current_prediction = torch.zeros_like(weighted_target)
    current_span_frac = 0.0
    current_min = None
    current_max = None

    column_norm_sq = weighted_design.pow(2).sum(dim=0).clamp_min(1e-12)
    normalized_candidate_keys = torch.nn.functional.normalize(candidate_keys, dim=1)

    for _ in range(budget):
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
            new_span_frac = (candidate_ends - candidate_starts + 1).float() / max(n_tokens, 1)
        else:
            new_min = torch.minimum(candidate_starts, torch.full_like(candidate_starts, current_min))
            new_max = torch.maximum(candidate_ends, torch.full_like(candidate_ends, current_max))
            new_span_frac = (new_max - new_min + 1).float() / max(n_tokens, 1)
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
        current_span_frac = float(((current_max - current_min + 1) / max(n_tokens, 1)))

        selected_design = weighted_design[:, selected_indices]
        scale = torch.linalg.lstsq(
            selected_design,
            weighted_target.unsqueeze(1),
            driver="gels",
        ).solution.squeeze(1)
        scale = scale.clamp_min(1e-12)
        current_prediction = selected_design @ scale

    if not selected_indices:
        return fallback_rep

    device = target_keys.device
    dtype = target_keys.dtype
    index_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
    support_keys = candidate_keys[index_tensor].to(dtype=dtype, device=device)
    support_values = candidate_values[index_tensor].to(dtype=candidate_values.dtype, device=candidate_values.device)
    selected_design = weighted_design[:, selected_indices]
    betas = torch.linalg.lstsq(
        selected_design,
        weighted_target.unsqueeze(1),
        driver="gels",
    ).solution.squeeze(1).clamp_min(1e-12).to(dtype=dtype, device=device)
    return CompactRepresentation(
        support_keys=support_keys,
        support_values=support_values,
        betas=betas,
    )


def _fit_adjacent_pair_representative(
    *,
    query_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
    left_key: torch.Tensor,
    right_key: torch.Tensor,
    left_value: torch.Tensor,
    right_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a conservative local representative for one adjacent pair.

    The construction only searches over a tiny discrete family:
    - key candidates: left key, right key, arithmetic mean
    - value candidates: left value, right value, arithmetic mean

    For each key candidate, it fits a single nonnegative beta against the
    pair's local mass target. It then chooses the value candidate with the
    lowest local numerator error under that beta. This keeps the merge
    construction source-grounded and intentionally low-capacity.
    """
    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    left_logits = query_tensor @ left_key.unsqueeze(1) * inv_sqrt_d
    right_logits = query_tensor @ right_key.unsqueeze(1) * inv_sqrt_d
    pair_logits = torch.cat([left_logits, right_logits], dim=1)
    reference_max = pair_logits.max(dim=1, keepdim=True).values
    left_scores = torch.exp(left_logits - reference_max).squeeze(1)
    right_scores = torch.exp(right_logits - reference_max).squeeze(1)
    pair_target_z = left_scores + right_scores
    pair_target_n = (
        left_scores.unsqueeze(1) * left_value.unsqueeze(0)
        + right_scores.unsqueeze(1) * right_value.unsqueeze(0)
    )
    row_weights = torch.sqrt(torch.clamp_min(entry_weights.to(dtype=torch.float32), 0.0))
    weighted_target_z = pair_target_z * row_weights
    weighted_target_n = pair_target_n * row_weights.unsqueeze(1)
    dv = max(int(left_value.shape[0]), 1)

    key_candidates = [
        left_key,
        right_key,
        0.5 * (left_key + right_key),
    ]
    value_candidates = [
        left_value,
        right_value,
        0.5 * (left_value + right_value),
    ]

    best_loss = float("inf")
    best_key = key_candidates[0]
    best_value = value_candidates[0]
    for key_candidate in key_candidates:
        candidate_logits = (query_tensor @ key_candidate.unsqueeze(1) * inv_sqrt_d).squeeze(1)
        candidate_scores = torch.exp(candidate_logits - reference_max.squeeze(1))
        weighted_column = candidate_scores * row_weights
        beta_hat = float(
            torch.dot(weighted_column, weighted_target_z).item()
            / max(torch.dot(weighted_column, weighted_column).item(), 1e-12)
        )
        beta_hat = max(beta_hat, 1e-12)
        z_residual = beta_hat * weighted_column - weighted_target_z
        z_loss = float(torch.dot(z_residual, z_residual).item())

        for value_candidate in value_candidates:
            pred_n = beta_hat * candidate_scores.unsqueeze(1) * value_candidate.unsqueeze(0)
            weighted_pred_n = pred_n * row_weights.unsqueeze(1)
            n_loss = float((weighted_pred_n - weighted_target_n).pow(2).sum().item()) / dv
            total_loss = z_loss + n_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_key = key_candidate
                best_value = value_candidate

    return best_key, best_value


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
