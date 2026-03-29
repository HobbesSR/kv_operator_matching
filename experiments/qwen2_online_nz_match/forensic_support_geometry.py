#!/usr/bin/env python3
"""
Comparative geometry / repair forensics for recency+vfit versus attn_mass+vfit.

This script is meant to explain the Phase 2 result that anchored value repair
works well on recency supports under decode-like evidence but remains weak on
attention-mass supports outside supervision-rich control regimes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import attention_mass_baseline, recency_baseline
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_logits, compute_response, compute_z, loss_n
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file


@dataclass
class GeometryRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    support_method: str
    train_queries: int
    holdout_queries: int
    baseline_l_true: float
    vfit_l_true: float
    delta_holdout_l_true: float
    baseline_l_z: float
    vfit_l_z: float
    baseline_l_n_per_dim: float
    vfit_l_n_per_dim: float
    design_rank: int
    design_stable_rank: float
    design_condition_number: float
    mean_support_age: float
    mean_support_age_frac: float
    support_span: int
    support_span_frac: float
    support_adjacent_fraction: float
    update_norm_over_base: float
    delta_over_base_mean: float
    delta_max: float
    delta_top5_share: float
    low_sv_delta_share: float
    holdout_top5_error_share: float
    holdout_median_query_error: float
    holdout_max_query_error: float


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
    p.add_argument("--continuation-turns", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/forensic_support_geometry.json")
    return p.parse_args()


def compute_metrics(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank):
    queries, weights = qbank.get_weighted_bank()
    total_w = weights.sum().clamp(min=1e-12)
    l_true = (
        ((compute_response(queries, rep.support_keys, rep.support_values, rep.betas) -
          compute_response(queries, head_state.keys, head_state.values)) ** 2).sum(dim=-1) * weights
    ).sum() / total_w
    l_z = ((((compute_z(queries, rep.support_keys, rep.betas) -
               compute_z(queries, head_state.keys)) ** 2) * weights).sum() / total_w)
    l_n = loss_n(
        queries,
        weights,
        rep.support_keys,
        rep.support_values,
        rep.betas,
        head_state.keys,
        head_state.values,
    ) / (total_w * head_state.values.shape[-1])
    return float(l_true.item()), float(l_z.item()), float(l_n.item())


def per_query_error_stats(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank):
    queries, weights = qbank.get_weighted_bank()
    a_hat = compute_response(queries, rep.support_keys, rep.support_values, rep.betas)
    a_ref = compute_response(queries, head_state.keys, head_state.values)
    per_q = ((a_hat - a_ref) ** 2).sum(dim=-1)
    weighted = per_q * weights
    total = weighted.sum().clamp(min=1e-12)
    topk = min(5, weighted.numel())
    return {
        "top5_share": float(weighted.topk(topk).values.sum().item() / total.item()),
        "median": float(per_q.median().item()),
        "max": float(per_q.max().item()),
    }


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
        "rank": int(torch.linalg.matrix_rank(design).item()),
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
    """
    Map each support key back to a source token index in the full cache.

    Supports are selected directly from the cache, so exact rows should exist.
    Use a nearest-row fallback to avoid brittle equality assumptions.
    """
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
        "mean_age": float(ages.mean().item()),
        "mean_age_frac": float((ages.mean() / max(n_tokens - 1, 1)).item()),
        "span": span,
        "span_frac": float(span / max(n_tokens, 1)),
        "adjacent_fraction": adjacent_fraction,
    }


def delta_stats(base_rep: CompactRepresentation, new_rep: CompactRepresentation, design: torch.Tensor) -> dict:
    base_norm = torch.linalg.vector_norm(base_rep.support_values, dim=1)
    new_norm = torch.linalg.vector_norm(new_rep.support_values, dim=1)
    delta = new_rep.support_values - base_rep.support_values
    delta_norm = torch.linalg.vector_norm(delta, dim=1)
    topk = min(5, delta_norm.numel())
    total_delta = delta_norm.sum().clamp(min=1e-12)
    return {
        "update_norm_over_base": float(new_norm.mean().item() / base_norm.mean().clamp(min=1e-12).item()),
        "delta_over_base_mean": float((delta_norm / base_norm.clamp(min=1e-12)).mean().item()),
        "delta_max": float(delta_norm.max().item()),
        "delta_top5_share": float(delta_norm.topk(topk).values.sum().item() / total_delta.item()),
        "low_sv_delta_share": low_singular_delta_share(design, delta),
    }


def summarize_rows(rows: List[GeometryRow]) -> dict:
    summary: dict[str, list] = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.collection_mode, row.support_method)].append(row)
    for (mode, support_method), group in sorted(grouped.items()):
        summary_key = f"{mode}:{support_method}"
        summary[summary_key] = {
            "num_rows": len(group),
            "mean_delta_holdout_l_true": sum(r.delta_holdout_l_true for r in group) / len(group),
            "improved_rows": sum(r.delta_holdout_l_true < 0 for r in group),
            "mean_design_stable_rank": sum(r.design_stable_rank for r in group) / len(group),
            "mean_design_condition_number": sum(r.design_condition_number for r in group) / len(group),
            "mean_support_age_frac": sum(r.mean_support_age_frac for r in group) / len(group),
            "mean_support_span_frac": sum(r.support_span_frac for r in group) / len(group),
            "mean_support_adjacent_fraction": sum(r.support_adjacent_fraction for r in group) / len(group),
            "mean_delta_over_base": sum(r.delta_over_base_mean for r in group) / len(group),
            "mean_low_sv_delta_share": sum(r.low_sv_delta_share for r in group) / len(group),
            "mean_holdout_top5_error_share": sum(r.holdout_top5_error_share for r in group) / len(group),
        }
    return summary


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
    rows: List[GeometryRow] = []
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads

    for prompt_file in args.prompt_files:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = smoke_data_path / prompt_file
        prompt, continuation_text = load_prompt_segments_from_file(
            prompt_path,
            prefix_turns=args.prefix_turns,
            continuation_turns=args.continuation_turns,
        )

        for collection_mode in args.collection_modes:
            print(f"\n=== prompt={prompt_path.name} mode={collection_mode} ===", flush=True)
            try:
                kv_states, query_banks, _meta = collect_mode_state(
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
                continue

            for layer_idx in args.layers:
                seq_len = kv_states[layer_idx]["keys"].shape[1]
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

                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        for support_method in ("recency", "attn_mass"):
                            if support_method == "recency":
                                base_rep = recency_baseline(head_state, budget)
                            else:
                                base_rep = attention_mass_baseline(head_state, train_qbank, budget)

                            vfit_rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                            train_queries, train_weights = train_qbank.get_weighted_bank()
                            design = build_design(
                                train_queries,
                                train_weights,
                                base_rep.support_keys,
                                base_rep.betas,
                            )
                            d_stats = matrix_stats(design)
                            g_stats = support_geometry(keys, base_rep.support_keys)
                            upd_stats = delta_stats(base_rep, vfit_rep, design)
                            base_l_true, base_l_z, base_l_n = compute_metrics(base_rep, head_state, holdout_qbank)
                            vfit_l_true, vfit_l_z, vfit_l_n = compute_metrics(vfit_rep, head_state, holdout_qbank)
                            err_stats = per_query_error_stats(vfit_rep, head_state, holdout_qbank)

                            rows.append(
                                GeometryRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    support_method=support_method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
                                    baseline_l_true=base_l_true,
                                    vfit_l_true=vfit_l_true,
                                    delta_holdout_l_true=vfit_l_true - base_l_true,
                                    baseline_l_z=base_l_z,
                                    vfit_l_z=vfit_l_z,
                                    baseline_l_n_per_dim=base_l_n,
                                    vfit_l_n_per_dim=vfit_l_n,
                                    design_rank=d_stats["rank"],
                                    design_stable_rank=d_stats["stable_rank"],
                                    design_condition_number=d_stats["cond"],
                                    mean_support_age=g_stats["mean_age"],
                                    mean_support_age_frac=g_stats["mean_age_frac"],
                                    support_span=g_stats["span"],
                                    support_span_frac=g_stats["span_frac"],
                                    support_adjacent_fraction=g_stats["adjacent_fraction"],
                                    update_norm_over_base=upd_stats["update_norm_over_base"],
                                    delta_over_base_mean=upd_stats["delta_over_base_mean"],
                                    delta_max=upd_stats["delta_max"],
                                    delta_top5_share=upd_stats["delta_top5_share"],
                                    low_sv_delta_share=upd_stats["low_sv_delta_share"],
                                    holdout_top5_error_share=err_stats["top5_share"],
                                    holdout_median_query_error=err_stats["median"],
                                    holdout_max_query_error=err_stats["max"],
                                )
                            )

    summary = summarize_rows(rows)
    for key, stats in summary.items():
        print(f"\n{key}")
        for metric, value in stats.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "rows": [asdict(r) for r in rows],
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nSaved forensic geometry report to {out}")


if __name__ == "__main__":
    main()
