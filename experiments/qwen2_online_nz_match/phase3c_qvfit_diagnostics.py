#!/usr/bin/env python3
"""
Phase 3C mechanism diagnostics for quotient-aware fixed-support refit.

This script asks what predicts qvfit-vfit improvement on a fixed support.
It compares ordinary vfit and quotient-aware qvfit on the same support family,
then records support-side geometry and query-weighting diagnostics that may
explain when qvfit helps or destabilizes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import (  # noqa: E402
    attention_mass_baseline,
    hybrid_support_baseline,
    omp_mass_baseline,
    quotient_omit_omp_baseline,
)
from kv_operator_matching.config import BetaFitConfig  # noqa: E402
from kv_operator_matching.objectives import (  # noqa: E402
    compute_logits,
    compute_quotient_residual_diagnostics,
    compute_response,
)
from kv_operator_matching.query_bank import QueryBank  # noqa: E402
from kv_operator_matching.types import CompactRepresentation, HeadState  # noqa: E402
from kv_operator_matching.value_fit import refit_values, refit_values_quotient  # noqa: E402

from compare_collection_modes import collect_mode_state  # noqa: E402
from run_experiment import load_model, load_prompt_segments_from_file  # noqa: E402


@dataclass
class QvfitDiagRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    base_method: str
    train_queries: int
    holdout_queries: int
    base_holdout_l_true: float
    vfit_holdout_l_true: float
    qvfit_holdout_l_true: float
    qvfit_minus_vfit_l_true: float
    base_train_qr_per_dim: float
    base_train_qr_cancellation_gain: float
    base_train_qr_worst_ratio: float
    alpha_design_stable_rank: float
    alpha_design_condition_number: float
    q_design_stable_rank: float
    q_design_condition_number: float
    q_row_energy_top5_share: float
    q_row_energy_cv: float
    zhat_over_zref_mean: float
    zhat_over_zref_cv: float
    beta_mean: float
    beta_max: float
    beta_cv: float


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", nargs="+", type=int, default=[4, 20])
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument(
        "--collection-modes",
        nargs="+",
        default=["online", "teacher-forced-suffix", "repeat-prefill"],
    )
    p.add_argument(
        "--prompt-files",
        nargs="+",
        default=[
            "near_capacity_dispatch_safe.json",
            "relational_binding_probe.json",
        ],
    )
    p.add_argument("--prefix-turns", type=int, default=8)
    p.add_argument("--continuation-turns", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument(
        "--base-methods",
        nargs="+",
        default=["attn_mass", "omp", "hybrid", "quotient_omit_omp"],
    )
    p.add_argument("--shortlist-multiplier", type=float, default=2.0)
    p.add_argument(
        "--quotient-score-mode",
        choices=["exact_local", "proxy"],
        default="exact_local",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3c_qvfit_diagnostics.json")
    return p.parse_args()


def compute_l_true(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank) -> float:
    queries, weights = qbank.get_weighted_bank()
    total_w = weights.sum().clamp(min=1e-12)
    l_true = (
        (
            (
                compute_response(queries, rep.support_keys, rep.support_values, rep.betas)
                - compute_response(queries, head_state.keys, head_state.values)
            )
            ** 2
        ).sum(dim=-1)
        * weights
    ).sum() / total_w
    return float(l_true.item())


def quotient_stats(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank) -> dict[str, float]:
    queries, weights = qbank.get_weighted_bank()
    total_w = weights.sum().clamp(min=1e-12)
    terms = compute_quotient_residual_diagnostics(
        queries,
        rep.support_keys,
        rep.support_values,
        rep.betas,
        head_state.keys,
        head_state.values,
    )
    d_v = head_state.values.shape[-1]
    delta_n_sq = terms["delta_n"].square().sum(dim=-1)
    o_delta_z_sq = (terms["a_ref"] * terms["delta_z"].unsqueeze(-1)).square().sum(dim=-1)
    qr_sq = terms["quotient_residual"].square().sum(dim=-1)
    cancellation_gain = 1.0 - qr_sq / (delta_n_sq + o_delta_z_sq).clamp(min=1e-12)
    qr_ratio = qr_sq.sqrt() / terms["z_ref"].clamp(min=1e-12)
    return {
        "qr_per_dim": float(((weights * qr_sq).sum() / (total_w * d_v)).item()),
        "qr_cancellation_gain": float(((weights * cancellation_gain).sum() / total_w).item()),
        "qr_worst_ratio": float(qr_ratio.max().item()),
    }


def matrix_stats(design: torch.Tensor) -> dict[str, float]:
    sv = torch.linalg.svdvals(design)
    smax = float(sv.max().item())
    smin = float(sv.min().item())
    return {
        "cond": smax / max(smin, 1e-12),
        "stable_rank": float((design.square().sum() / max(smax * smax, 1e-12)).item()),
    }


def qvfit_predictors(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank) -> dict[str, float]:
    queries, weights = qbank.get_weighted_bank()
    queries = queries.float()
    weights = weights.float()
    support_keys = rep.support_keys.float()
    betas = rep.betas.float()
    ref_keys = head_state.keys.float()

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
    alpha = exp_hat / z_hat.unsqueeze(-1)

    sqrt_w = weights.sqrt().unsqueeze(-1)
    alpha_design = alpha * sqrt_w
    q_design = exp_hat * sqrt_w
    alpha_stats = matrix_stats(alpha_design)
    q_stats = matrix_stats(q_design)

    row_energy = weights * z_hat.square()
    row_energy_total = row_energy.sum().clamp(min=1e-12)
    topk = min(5, int(row_energy.numel()))
    top5_share = float(row_energy.topk(topk).values.sum().item() / row_energy_total.item())
    row_mean = float(row_energy.mean().item())
    row_std = float(row_energy.std(unbiased=False).item())
    row_cv = row_std / max(row_mean, 1e-12)

    z_ratio = z_hat / z_ref
    z_ratio_mean = float((weights * z_ratio).sum().item() / weights.sum().clamp(min=1e-12).item())
    z_ratio_centered = z_ratio - z_ratio_mean
    z_ratio_var = float((weights * z_ratio_centered.square()).sum().item() / weights.sum().clamp(min=1e-12).item())
    z_ratio_cv = math.sqrt(max(z_ratio_var, 0.0)) / max(z_ratio_mean, 1e-12)

    beta_mean = float(betas.mean().item())
    beta_std = float(betas.std(unbiased=False).item())
    return {
        "alpha_design_stable_rank": alpha_stats["stable_rank"],
        "alpha_design_condition_number": alpha_stats["cond"],
        "q_design_stable_rank": q_stats["stable_rank"],
        "q_design_condition_number": q_stats["cond"],
        "q_row_energy_top5_share": top5_share,
        "q_row_energy_cv": row_cv,
        "zhat_over_zref_mean": z_ratio_mean,
        "zhat_over_zref_cv": z_ratio_cv,
        "beta_mean": beta_mean,
        "beta_max": float(betas.max().item()),
        "beta_cv": beta_std / max(beta_mean, 1e-12),
    }


def summarize(rows: List[QvfitDiagRow]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[QvfitDiagRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.collection_mode, row.base_method)].append(row)

    summary = {}
    for key, group in sorted(grouped.items()):
        summary[f"{key[0]}::{key[1]}"] = {
            "cells": float(len(group)),
            "mean_qvfit_minus_vfit_l_true": sum(r.qvfit_minus_vfit_l_true for r in group) / len(group),
            "mean_alpha_design_condition_number": sum(r.alpha_design_condition_number for r in group) / len(group),
            "mean_q_design_condition_number": sum(r.q_design_condition_number for r in group) / len(group),
            "mean_q_row_energy_top5_share": sum(r.q_row_energy_top5_share for r in group) / len(group),
            "mean_zhat_over_zref_cv": sum(r.zhat_over_zref_cv for r in group) / len(group),
            "mean_base_train_qr_cancellation_gain": sum(r.base_train_qr_cancellation_gain for r in group) / len(group),
        }
    return summary


def correlation(rows: List[QvfitDiagRow], x_field: str, y_field: str) -> float:
    xs = torch.tensor([getattr(r, x_field) for r in rows], dtype=torch.float64)
    ys = torch.tensor([getattr(r, y_field) for r in rows], dtype=torch.float64)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = xs.norm() * ys.norm()
    if denom.item() == 0:
        return 0.0
    return float((xs @ ys / denom).item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)

    beta_cfg = BetaFitConfig(
        support_size=64,
        max_iter=0,
        tol=1e-6,
        nonneg=True,
        surrogate="lin",
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
    )

    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    rows: List[QvfitDiagRow] = []
    n_kv_heads = model.config.num_key_value_heads
    exact_local = args.quotient_score_mode == "exact_local"

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
                        base_reps: Dict[str, CompactRepresentation] = {}
                        if "attn_mass" in args.base_methods:
                            base_reps["attn_mass"] = attention_mass_baseline(head_state, train_qbank, budget)
                        if "omp" in args.base_methods:
                            base_reps["omp"] = omp_mass_baseline(head_state, train_qbank, budget)
                        if "hybrid" in args.base_methods:
                            base_reps["hybrid"] = hybrid_support_baseline(head_state, train_qbank, budget)
                        if "quotient_omit_omp" in args.base_methods:
                            base_reps["quotient_omit_omp"] = quotient_omit_omp_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                shortlist_multiplier=args.shortlist_multiplier,
                                exact_local=exact_local,
                            )

                        for base_method, base_rep in base_reps.items():
                            vfit_rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                            qvfit_rep = refit_values_quotient(base_rep, keys, values, train_qbank, beta_cfg)
                            predictors = qvfit_predictors(base_rep, head_state, train_qbank)
                            base_qstats = quotient_stats(base_rep, head_state, train_qbank)
                            rows.append(
                                QvfitDiagRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    base_method=base_method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
                                    base_holdout_l_true=compute_l_true(base_rep, head_state, holdout_qbank),
                                    vfit_holdout_l_true=compute_l_true(vfit_rep, head_state, holdout_qbank),
                                    qvfit_holdout_l_true=compute_l_true(qvfit_rep, head_state, holdout_qbank),
                                    qvfit_minus_vfit_l_true=(
                                        compute_l_true(qvfit_rep, head_state, holdout_qbank)
                                        - compute_l_true(vfit_rep, head_state, holdout_qbank)
                                    ),
                                    base_train_qr_per_dim=base_qstats["qr_per_dim"],
                                    base_train_qr_cancellation_gain=base_qstats["qr_cancellation_gain"],
                                    base_train_qr_worst_ratio=base_qstats["qr_worst_ratio"],
                                    alpha_design_stable_rank=predictors["alpha_design_stable_rank"],
                                    alpha_design_condition_number=predictors["alpha_design_condition_number"],
                                    q_design_stable_rank=predictors["q_design_stable_rank"],
                                    q_design_condition_number=predictors["q_design_condition_number"],
                                    q_row_energy_top5_share=predictors["q_row_energy_top5_share"],
                                    q_row_energy_cv=predictors["q_row_energy_cv"],
                                    zhat_over_zref_mean=predictors["zhat_over_zref_mean"],
                                    zhat_over_zref_cv=predictors["zhat_over_zref_cv"],
                                    beta_mean=predictors["beta_mean"],
                                    beta_max=predictors["beta_max"],
                                    beta_cv=predictors["beta_cv"],
                                )
                            )

    summary = summarize(rows)
    print("\nGrouped summary:")
    for key, stats in summary.items():
        print(f"\n{key}")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.4f}")

    print("\nGlobal correlations with qvfit_minus_vfit_l_true:")
    for field in [
        "alpha_design_condition_number",
        "q_design_condition_number",
        "q_row_energy_top5_share",
        "q_row_energy_cv",
        "zhat_over_zref_cv",
        "base_train_qr_cancellation_gain",
        "base_train_qr_worst_ratio",
        "beta_cv",
    ]:
        print(f"  {field}: {correlation(rows, field, 'qvfit_minus_vfit_l_true'):.4f}")

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "rows": [asdict(r) for r in rows],
                "summary": summary,
                "correlations": {
                    field: correlation(rows, field, "qvfit_minus_vfit_l_true")
                    for field in [
                        "alpha_design_condition_number",
                        "q_design_condition_number",
                        "q_row_energy_top5_share",
                        "q_row_energy_cv",
                        "zhat_over_zref_cv",
                        "base_train_qr_cancellation_gain",
                        "base_train_qr_worst_ratio",
                        "beta_cv",
                    ]
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved Phase 3C qvfit diagnostics to {out}")


if __name__ == "__main__":
    main()
