#!/usr/bin/env python3
"""
Phase 3A ablations and support-geometry forensics.

This script answers three narrow questions before Phase 3B:

1. Which hybrid-score terms are carrying the Phase 3A win?
2. Does hybrid support geometry really sit between recency and OMP in the way
   suggested by the Phase 2 tradeoff story?
3. Does the result survive a small stress surface beyond the original checkpoint?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import (
    HybridSelectorConfig,
    hybrid_evidence_weights,
    hybrid_support_baseline,
    omp_mass_baseline,
    recency_baseline,
)
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_logits, compute_response, compute_z, loss_n
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file


@dataclass
class AblationRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    baseline_method: str
    train_queries: int
    holdout_queries: int
    alpha: float
    beta: float
    baseline_l_true: float
    vfit_l_true: float
    delta_holdout_l_true: float


@dataclass
class GeometryRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    train_queries: int
    holdout_queries: int
    baseline_l_true: float
    vfit_l_true: float
    design_stable_rank: float
    design_condition_number: float
    support_span_frac: float
    support_adjacent_fraction: float
    mean_support_age_frac: float
    low_sv_delta_share: float


VARIANT_CONFIGS = {
    "hybrid": HybridSelectorConfig(),
    "hybrid_db_only": HybridSelectorConfig(
        use_delta_b=True,
        use_delta_q_coh=False,
        use_delta_q_span=False,
    ),
    "hybrid_db_coh": HybridSelectorConfig(
        use_delta_b=True,
        use_delta_q_coh=True,
        use_delta_q_span=False,
    ),
    "hybrid_db_coh_lowsv": HybridSelectorConfig(
        use_delta_b=True,
        use_delta_q_coh=True,
        use_delta_q_span=False,
        use_delta_q_low_sv_risk=True,
    ),
    "hybrid_db_span": HybridSelectorConfig(
        use_delta_b=True,
        use_delta_q_coh=False,
        use_delta_q_span=True,
    ),
    "hybrid_unweighted": HybridSelectorConfig(
        use_delta_b=True,
        use_delta_q_coh=True,
        use_delta_q_span=True,
        use_evidence_weights=False,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28])
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument(
        "--collection-modes",
        nargs="+",
        default=["online", "teacher-forced", "repeat-prefill"],
    )
    p.add_argument(
        "--prompt-files",
        nargs="+",
        default=[
            "near_capacity_dispatch.json",
            "near_capacity_dispatch_safe.json",
            "near_capacity_network_dispatch_safe.json",
            "relational_binding_probe.json",
        ],
    )
    p.add_argument("--prefix-turns", type=int, default=8)
    p.add_argument("--continuation-turns", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3a_ablation_forensics.json")
    return p.parse_args()


def compute_metrics(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank):
    queries, weights = qbank.get_weighted_bank()
    total_w = weights.sum().clamp(min=1e-12)
    l_true = (
        ((compute_response(queries, rep.support_keys, rep.support_values, rep.betas) -
          compute_response(queries, head_state.keys, head_state.values)) ** 2).sum(dim=-1) * weights
    ).sum() / total_w
    return float(l_true.item())


def build_design(
    queries: torch.Tensor,
    weights: torch.Tensor,
    support_keys: torch.Tensor,
    betas: torch.Tensor,
):
    logits = compute_logits(queries, support_keys) + torch.log(betas.clamp(min=1e-30)).unsqueeze(0)
    alpha = torch.softmax(logits, dim=-1)
    return alpha * weights.sqrt().unsqueeze(-1)


def matrix_stats(design: torch.Tensor):
    sv = torch.linalg.svdvals(design)
    smax = float(sv.max().item())
    smin = float(sv.min().item())
    return {
        "cond": smax / max(smin, 1e-12),
        "stable_rank": float((design.square().sum() / max(smax * smax, 1e-12)).item()),
    }


def low_singular_delta_share(design: torch.Tensor, delta_values: torch.Tensor) -> float:
    _u, _s, vh = torch.linalg.svd(design, full_matrices=False)
    coeff = vh @ delta_values
    energy = coeff.pow(2).sum(dim=1)
    if energy.numel() == 0:
        return 0.0
    cutoff = max(1, energy.numel() // 4)
    low_energy = energy[-cutoff:].sum()
    return float((low_energy / energy.sum().clamp(min=1e-12)).item())


def match_support_indices(full_keys: torch.Tensor, support_keys: torch.Tensor) -> torch.Tensor:
    diffs = (support_keys.unsqueeze(1).float() - full_keys.unsqueeze(0).float()).pow(2).sum(dim=-1)
    return diffs.argmin(dim=1)


def support_geometry(full_keys: torch.Tensor, support_keys: torch.Tensor) -> dict:
    indices = match_support_indices(full_keys, support_keys).sort().values
    n_tokens = int(full_keys.shape[0])
    ages = (n_tokens - 1 - indices).float()
    if indices.numel() > 1:
        adjacent_fraction = float(((indices[1:] - indices[:-1]) == 1).float().mean().item())
        span = int((indices[-1] - indices[0]).item() + 1)
    else:
        adjacent_fraction = 1.0
        span = 1
    return {
        "mean_age_frac": float((ages.mean() / max(n_tokens - 1, 1)).item()),
        "span_frac": float(span / max(n_tokens, 1)),
        "adjacent_fraction": adjacent_fraction,
    }


def select_support(
    method: str,
    head_state: HeadState,
    train_qbank: QueryBank,
    config: HybridSelectorConfig | None = None,
) -> CompactRepresentation:
    budget = getattr(config, "budget", None)
    raise RuntimeError("select_support should not be called with budget embedded in config")


def build_rep(
    method: str,
    head_state: HeadState,
    train_qbank: QueryBank,
    budget: int,
    frozen_online_weights: tuple[float, float] | None,
) -> tuple[CompactRepresentation, float, float]:
    if method == "recency":
        return recency_baseline(head_state, budget), 0.0, 0.0
    if method == "omp":
        return omp_mass_baseline(head_state, train_qbank, budget), 0.0, 0.0
    if method == "hybrid_frozen_online":
        alpha, beta = frozen_online_weights if frozen_online_weights is not None else (1.0, 1.0)
        cfg = HybridSelectorConfig(
            use_delta_b=True,
            use_delta_q_coh=True,
            use_delta_q_span=True,
            use_evidence_weights=False,
            fixed_alpha=alpha,
            fixed_beta=beta,
        )
        return hybrid_support_baseline(head_state, train_qbank, budget, cfg), alpha, beta
    if method in VARIANT_CONFIGS:
        cfg = VARIANT_CONFIGS[method]
        queries, weights = train_qbank.get_weighted_bank()
        alpha, beta = hybrid_evidence_weights(
            queries.float(),
            weights.float(),
            use_evidence_weights=cfg.use_evidence_weights,
            fixed_alpha=cfg.fixed_alpha,
            fixed_beta=cfg.fixed_beta,
        )
        return hybrid_support_baseline(head_state, train_qbank, budget, cfg), alpha, beta
    raise ValueError(f"Unsupported method: {method}")


def summarize_ablation(rows: List[AblationRow]) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[(row.collection_mode, row.method)].append(row)
    summary = {}
    for key, group in sorted(by.items()):
        mode, method = key
        summary[f"{mode}:{method}"] = {
            "num_rows": len(group),
            "baseline": group[0].baseline_method,
            "mean_delta_holdout_l_true": sum(r.delta_holdout_l_true for r in group) / len(group),
            "improved_rows": sum(r.delta_holdout_l_true < 0 for r in group),
            "mean_alpha": sum(r.alpha for r in group) / len(group),
            "mean_beta": sum(r.beta for r in group) / len(group),
            "mean_holdout_l_true": sum(r.vfit_l_true for r in group) / len(group),
        }
    return summary


def summarize_geometry(rows: List[GeometryRow]) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[(row.collection_mode, row.method)].append(row)
    summary = {}
    for key, group in sorted(by.items()):
        mode, method = key
        summary[f"{mode}:{method}"] = {
            "num_rows": len(group),
            "mean_baseline_l_true": sum(r.baseline_l_true for r in group) / len(group),
            "mean_vfit_l_true": sum(r.vfit_l_true for r in group) / len(group),
            "mean_design_stable_rank": sum(r.design_stable_rank for r in group) / len(group),
            "mean_design_condition_number": sum(r.design_condition_number for r in group) / len(group),
            "mean_support_span_frac": sum(r.support_span_frac for r in group) / len(group),
            "mean_support_adjacent_fraction": sum(r.support_adjacent_fraction for r in group) / len(group),
            "mean_support_age_frac": sum(r.mean_support_age_frac for r in group) / len(group),
            "mean_low_sv_delta_share": sum(r.low_sv_delta_share for r in group) / len(group),
        }
    return summary


def summarize_pairwise(rows: List[AblationRow], methods: List[str]) -> dict:
    by = {}
    for row in rows:
        by[(row.prompt_id, row.collection_mode, row.layer, row.kv_head, row.budget, row.method)] = row.vfit_l_true
    out = {}
    pairs = [
        ("hybrid", "recency"),
        ("hybrid", "omp"),
        ("hybrid", "hybrid_db_only"),
        ("hybrid", "hybrid_db_coh"),
        ("hybrid", "hybrid_db_coh_lowsv"),
        ("hybrid", "hybrid_db_span"),
        ("hybrid", "hybrid_unweighted"),
        ("hybrid", "hybrid_frozen_online"),
        ("hybrid_db_coh_lowsv", "hybrid_db_coh"),
    ]
    for left, right in pairs:
        diffs_by_mode = defaultdict(list)
        for key, value in by.items():
            prompt_id, mode, layer, kv_head, budget, method = key
            if method != left:
                continue
            other_key = (prompt_id, mode, layer, kv_head, budget, right)
            if other_key in by:
                diffs_by_mode[mode].append(value - by[other_key])
        for mode, diffs in sorted(diffs_by_mode.items()):
            out[f"{mode}:{left}_minus_{right}"] = {
                "num_rows": len(diffs),
                "mean_difference": sum(diffs) / len(diffs),
                "left_better_rows": sum(d < 0 for d in diffs),
            }
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)

    beta_cfg = BetaFitConfig(
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
    )

    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    n_kv_heads = model.config.num_key_value_heads

    ablation_methods = [
        "recency",
        "omp",
        "hybrid",
        "hybrid_db_only",
        "hybrid_db_coh",
        "hybrid_db_coh_lowsv",
        "hybrid_db_span",
        "hybrid_unweighted",
        "hybrid_frozen_online",
    ]
    geometry_methods = ["recency", "omp", "hybrid"]

    ablation_rows: List[AblationRow] = []
    geometry_rows: List[GeometryRow] = []

    for prompt_file in args.prompt_files:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = smoke_data_path / prompt_file
        prompt, continuation_text = load_prompt_segments_from_file(
            prompt_path,
            prefix_turns=args.prefix_turns,
            continuation_turns=args.continuation_turns,
        )

        mode_state = {}
        for collection_mode in args.collection_modes:
            print(f"\n=== prompt={prompt_path.name} mode={collection_mode} ===", flush=True)
            try:
                mode_state[collection_mode] = collect_mode_state(
                    mode=collection_mode,
                    prompt=prompt,
                    continuation_text=continuation_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    layers=args.layers,
                    max_queries=args.max_queries,
                    max_new_tokens=args.max_new_tokens,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            except RuntimeError as exc:
                print(f"  skipped: {exc}", flush=True)

        for collection_mode, (kv_states, query_banks, _meta) in mode_state.items():
            online_state = mode_state.get("online")
            for layer_idx in args.layers:
                seq_len = kv_states[layer_idx]["keys"].shape[1]
                online_kv_states, online_query_banks, _ = online_state if online_state is not None else (None, None, None)
                for kv_head in range(n_kv_heads):
                    qbank = query_banks[(layer_idx, kv_head)]
                    train_qbank, holdout_qbank = qbank.split_train_holdout(args.train_fraction)
                    keys = kv_states[layer_idx]["keys"][kv_head]
                    values = kv_states[layer_idx]["values"][kv_head]
                    head_state = HeadState(
                        head_idx=kv_head,
                        layer_idx=layer_idx,
                        keys=keys,
                        values=values,
                    )

                    online_frozen = None
                    if online_query_banks is not None:
                        online_train_qbank, _ = online_query_banks[(layer_idx, kv_head)].split_train_holdout(args.train_fraction)
                        online_queries, online_weights = online_train_qbank.get_weighted_bank()
                        online_frozen = hybrid_evidence_weights(online_queries.float(), online_weights.float())

                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        cached = {}
                        for method in ablation_methods:
                            base_rep, alpha, beta = build_rep(
                                method,
                                head_state,
                                train_qbank,
                                budget,
                                online_frozen,
                            )
                            vfit_rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                            base_l_true = compute_metrics(base_rep, head_state, holdout_qbank)
                            vfit_l_true = compute_metrics(vfit_rep, head_state, holdout_qbank)
                            baseline_method = method
                            cached[method] = (base_rep, vfit_rep, alpha, beta, base_l_true, vfit_l_true)
                            ablation_rows.append(
                                AblationRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    method=f"{method}+vfit",
                                    baseline_method=baseline_method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
                                    alpha=alpha,
                                    beta=beta,
                                    baseline_l_true=base_l_true,
                                    vfit_l_true=vfit_l_true,
                                    delta_holdout_l_true=vfit_l_true - base_l_true,
                                )
                            )

                        train_queries, train_weights = train_qbank.get_weighted_bank()
                        for method in geometry_methods:
                            base_rep, vfit_rep, _alpha, _beta, base_l_true, vfit_l_true = cached[method]
                            design = build_design(
                                train_queries,
                                train_weights,
                                base_rep.support_keys,
                                base_rep.betas,
                            )
                            d_stats = matrix_stats(design)
                            g_stats = support_geometry(keys, base_rep.support_keys)
                            delta_values = vfit_rep.support_values - base_rep.support_values
                            geometry_rows.append(
                                GeometryRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    method=method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
                                    baseline_l_true=base_l_true,
                                    vfit_l_true=vfit_l_true,
                                    design_stable_rank=d_stats["stable_rank"],
                                    design_condition_number=d_stats["cond"],
                                    support_span_frac=g_stats["span_frac"],
                                    support_adjacent_fraction=g_stats["adjacent_fraction"],
                                    mean_support_age_frac=g_stats["mean_age_frac"],
                                    low_sv_delta_share=low_singular_delta_share(design, delta_values),
                                )
                            )

    output = {
        "args": vars(args),
        "ablation_rows": [asdict(r) for r in ablation_rows],
        "geometry_rows": [asdict(r) for r in geometry_rows],
        "ablation_summary": summarize_ablation(ablation_rows),
        "geometry_summary": summarize_geometry(geometry_rows),
        "pairwise_summary": summarize_pairwise(
            [r for r in ablation_rows if r.method.endswith("+vfit")],
            [r.method for r in ablation_rows],
        ),
    }

    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved Phase 3A ablation forensics to {out_path}")


if __name__ == "__main__":
    main()
