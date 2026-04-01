#!/usr/bin/env python3
"""
Phase 3C gated and tempered qvfit policy experiment.

This tranche converts the qvfit compatibility diagnostics into actual refit
policies on fixed supports:

- plain vfit
- raw qvfit
- hard-gated qvfit
- tempered qvfit
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

from kv_operator_matching.baselines import (  # noqa: E402
    attention_mass_baseline,
    hybrid_support_baseline,
    omp_mass_baseline,
    quotient_omit_omp_baseline,
    shortlist_omp_baseline,
)
from kv_operator_matching.config import BetaFitConfig  # noqa: E402
from kv_operator_matching.objectives import (  # noqa: E402
    compute_quotient_residual_diagnostics,
    compute_response,
)
from kv_operator_matching.query_bank import QueryBank  # noqa: E402
from kv_operator_matching.types import CompactRepresentation, HeadState  # noqa: E402
from kv_operator_matching.value_fit import (  # noqa: E402
    choose_diagnostic_qfit_row_scale_power,
    choose_qvfit_row_scale_power,
    compute_qvfit_row_scaling_stats,
    refit_values,
    refit_values_quotient,
    refit_values_quotient_gated,
)

from compare_collection_modes import collect_mode_state  # noqa: E402
from run_experiment import load_model, load_prompt_segments_from_file  # noqa: E402


@dataclass
class PolicyRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    train_queries: int
    holdout_queries: int
    train_zhat_over_zref_cv: float
    train_q_row_energy_top5_share: float
    selected_q_weight_neff_fraction: float
    selected_q_weight_kl_to_neutral: float
    q_metric_strength: float
    controller_branch: str
    used_qvfit: float
    holdout_l_true: float
    holdout_qr_per_dim: float
    holdout_qr_cancellation_gain: float
    holdout_qr_worst_ratio: float


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
        "--methods",
        nargs="+",
        default=[
            "attn_mass+vfit",
            "attn_mass+qvfit",
            "attn_mass+qvfit_gate",
            "attn_mass+qvfit_temp",
            "attn_mass+qvfit_neff",
            "attn_mass+qvfit_kl",
            "attn_mass+qfit_diag",
            "omp+vfit",
            "omp+qvfit",
            "omp+qvfit_gate",
            "omp+qvfit_temp",
            "omp+qvfit_neff",
            "omp+qvfit_kl",
            "omp+qfit_diag",
            "hybrid+vfit",
            "hybrid+qvfit",
            "hybrid+qvfit_gate",
            "hybrid+qvfit_temp",
            "hybrid+qvfit_neff",
            "hybrid+qvfit_kl",
            "hybrid+qfit_diag",
            "quotient_omit_omp+vfit",
            "quotient_omit_omp+qvfit",
            "quotient_omit_omp+qvfit_gate",
            "quotient_omit_omp+qvfit_temp",
            "quotient_omit_omp+qvfit_neff",
            "quotient_omit_omp+qvfit_kl",
            "quotient_omit_omp+qfit_diag",
            "rank_blend_omp+vfit",
            "rank_blend_omp+qfit_diag",
            "two_stage_gate_omp+vfit",
            "two_stage_gate_omp+qfit_diag",
        ],
    )
    p.add_argument("--shortlist-multiplier", type=float, default=2.0)
    p.add_argument("--gate-expansion", type=int, default=2)
    p.add_argument(
        "--quotient-score-mode",
        choices=["exact_local", "proxy"],
        default="exact_local",
    )
    p.add_argument("--gate-threshold", type=float, default=0.25)
    p.add_argument("--temp-power", type=float, default=0.5)
    p.add_argument("--neff-floor-fraction", type=float, default=0.5)
    p.add_argument("--max-kl-to-neutral", type=float, default=0.25)
    p.add_argument("--gamma-grid-size", type=int, default=65)
    p.add_argument("--controller-full-kl-threshold", type=float, default=0.9)
    p.add_argument("--controller-hard-gate-cv-threshold", type=float, default=0.5)
    p.add_argument("--controller-middle-control", choices=["kl", "neff"], default="kl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3c_qvfit_policy.json")
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


def summarize_rows(rows: List[PolicyRow]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[PolicyRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.collection_mode, row.method)].append(row)

    summary: dict[str, dict[str, float]] = {}
    for key, group in sorted(grouped.items()):
        deltas_vs_vfit = []
        for row in group:
            if row.method.endswith("+vfit"):
                continue
            peer_name = row.method.split("+", 1)[0] + "+vfit"
            peer = next(
                (
                    r
                    for r in rows
                    if r.prompt_id == row.prompt_id
                    and r.collection_mode == row.collection_mode
                    and r.layer == row.layer
                    and r.kv_head == row.kv_head
                    and r.budget == row.budget
                    and r.method == peer_name
                ),
                None,
            )
            if peer is not None:
                deltas_vs_vfit.append(row.holdout_l_true - peer.holdout_l_true)
        summary[f"{key[0]}::{key[1]}"] = {
            "cells": float(len(group)),
            "mean_holdout_l_true": sum(r.holdout_l_true for r in group) / len(group),
            "mean_holdout_qr_per_dim": sum(r.holdout_qr_per_dim for r in group) / len(group),
            "mean_holdout_qr_cancellation_gain": (
                sum(r.holdout_qr_cancellation_gain for r in group) / len(group)
            ),
            "mean_holdout_qr_worst_ratio": sum(r.holdout_qr_worst_ratio for r in group) / len(group),
            "mean_train_zhat_over_zref_cv": sum(r.train_zhat_over_zref_cv for r in group) / len(group),
            "mean_train_q_row_energy_top5_share": (
                sum(r.train_q_row_energy_top5_share for r in group) / len(group)
            ),
            "mean_selected_q_weight_neff_fraction": (
                sum(r.selected_q_weight_neff_fraction for r in group) / len(group)
            ),
            "mean_selected_q_weight_kl_to_neutral": (
                sum(r.selected_q_weight_kl_to_neutral for r in group) / len(group)
            ),
            "mean_q_metric_strength": sum(r.q_metric_strength for r in group) / len(group),
            "qvfit_use_rate": sum(r.used_qvfit for r in group) / len(group),
            "full_quotient_rate": sum(r.controller_branch == "full_quotient" for r in group) / len(group),
            "middle_kl_rate": sum(r.controller_branch == "middle_kl" for r in group) / len(group),
            "middle_neff_rate": sum(r.controller_branch == "middle_neff" for r in group) / len(group),
            "neutral_fallback_rate": sum(r.controller_branch == "neutral_fallback" for r in group) / len(group),
            "mean_delta_vs_vfit_peer": (sum(deltas_vs_vfit) / len(deltas_vs_vfit))
            if deltas_vs_vfit
            else 0.0,
        }
    return summary


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
    rows: List[PolicyRow] = []
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
                        if any(method.startswith("attn_mass") for method in args.methods):
                            base_reps["attn_mass"] = attention_mass_baseline(head_state, train_qbank, budget)
                        if any(method.startswith("omp") for method in args.methods):
                            base_reps["omp"] = omp_mass_baseline(head_state, train_qbank, budget)
                        if any(method.startswith("hybrid") for method in args.methods):
                            base_reps["hybrid"] = hybrid_support_baseline(head_state, train_qbank, budget)
                        if any(method.startswith("quotient_omit_omp") for method in args.methods):
                            base_reps["quotient_omit_omp"] = quotient_omit_omp_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                shortlist_multiplier=args.shortlist_multiplier,
                                exact_local=exact_local,
                            )
                        if any(method.startswith("rank_blend_omp") for method in args.methods):
                            base_reps["rank_blend_omp"] = shortlist_omp_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                shortlist_multiplier=args.shortlist_multiplier,
                                shortlist_policy="rank_blend",
                                exact_local=exact_local,
                                gate_expansion=args.gate_expansion,
                            )
                        if any(method.startswith("two_stage_gate_omp") for method in args.methods):
                            base_reps["two_stage_gate_omp"] = shortlist_omp_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                shortlist_multiplier=args.shortlist_multiplier,
                                shortlist_policy="two_stage_gate",
                                exact_local=exact_local,
                                gate_expansion=args.gate_expansion,
                            )

                        row_stats = {
                            name: compute_qvfit_row_scaling_stats(rep.support_keys, rep.betas, keys, train_qbank)
                            for name, rep in base_reps.items()
                        }
                        neff_controls = {}
                        if any(method.endswith("+qvfit_neff") for method in args.methods):
                            neff_controls = {
                                name: choose_qvfit_row_scale_power(
                                    rep.support_keys,
                                    rep.betas,
                                    keys,
                                    train_qbank,
                                    min_neff_fraction=args.neff_floor_fraction,
                                    grid_size=args.gamma_grid_size,
                                )
                                for name, rep in base_reps.items()
                            }
                        kl_controls = {}
                        if any(method.endswith("+qvfit_kl") for method in args.methods):
                            kl_controls = {
                                name: choose_qvfit_row_scale_power(
                                    rep.support_keys,
                                    rep.betas,
                                    keys,
                                    train_qbank,
                                    max_kl_to_neutral=args.max_kl_to_neutral,
                                    grid_size=args.gamma_grid_size,
                                )
                                for name, rep in base_reps.items()
                            }
                        diag_controls = {}
                        if any(method.endswith("+qfit_diag") for method in args.methods):
                            diag_controls = {
                                name: choose_diagnostic_qfit_row_scale_power(
                                    rep.support_keys,
                                    rep.betas,
                                    keys,
                                    train_qbank,
                                    full_metric_max_kl_to_neutral=args.controller_full_kl_threshold,
                                    hard_gate_zhat_over_zref_cv=args.controller_hard_gate_cv_threshold,
                                    middle_control=args.controller_middle_control,
                                    middle_min_neff_fraction=args.neff_floor_fraction,
                                    middle_max_kl_to_neutral=args.max_kl_to_neutral,
                                    grid_size=args.gamma_grid_size,
                                )
                                for name, rep in base_reps.items()
                            }

                        for method in args.methods:
                            base_name = method.split("+", 1)[0]
                            base_rep = base_reps[base_name]
                            stats = row_stats[base_name]
                            used_qvfit = 0.0
                            q_metric_strength = 0.0
                            controller_branch = "neutral_fallback"
                            selected_stats = compute_qvfit_row_scaling_stats(
                                base_rep.support_keys,
                                base_rep.betas,
                                keys,
                                train_qbank,
                                row_scale_power=0.0,
                            )
                            if method.endswith("+vfit"):
                                rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                            elif method.endswith("+qvfit"):
                                rep = refit_values_quotient(base_rep, keys, values, train_qbank, beta_cfg)
                                used_qvfit = 1.0
                                q_metric_strength = 1.0
                                controller_branch = "full_quotient"
                                selected_stats = stats
                            elif method.endswith("+qvfit_gate"):
                                gate_open = stats["zhat_over_zref_cv"] <= args.gate_threshold
                                rep = refit_values_quotient_gated(
                                    base_rep,
                                    keys,
                                    values,
                                    train_qbank,
                                    beta_cfg,
                                    zhat_over_zref_cv_threshold=args.gate_threshold,
                                )
                                used_qvfit = 1.0 if gate_open else 0.0
                                q_metric_strength = 1.0 if gate_open else 0.0
                                controller_branch = "full_quotient" if gate_open else "neutral_fallback"
                                selected_stats = stats if gate_open else compute_qvfit_row_scaling_stats(
                                    base_rep.support_keys,
                                    base_rep.betas,
                                    keys,
                                    train_qbank,
                                    row_scale_power=0.0,
                                )
                            elif method.endswith("+qvfit_temp"):
                                rep = refit_values_quotient(
                                    base_rep,
                                    keys,
                                    values,
                                    train_qbank,
                                    beta_cfg,
                                    row_scale_power=args.temp_power,
                                )
                                used_qvfit = 1.0
                                q_metric_strength = float(args.temp_power)
                                controller_branch = "middle_temp"
                                selected_stats = compute_qvfit_row_scaling_stats(
                                    base_rep.support_keys,
                                    base_rep.betas,
                                    keys,
                                    train_qbank,
                                    row_scale_power=args.temp_power,
                                )
                            elif method.endswith("+qvfit_neff"):
                                gamma, control_stats = neff_controls[base_name]
                                if gamma == 0.0:
                                    rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                                    used_qvfit = 0.0
                                else:
                                    rep = refit_values_quotient(
                                        base_rep,
                                        keys,
                                        values,
                                        train_qbank,
                                        beta_cfg,
                                        row_scale_power=gamma,
                                    )
                                    used_qvfit = 1.0
                                q_metric_strength = gamma
                                controller_branch = "neutral_fallback" if gamma == 0.0 else "middle_neff"
                                selected_stats = control_stats
                            elif method.endswith("+qvfit_kl"):
                                gamma, control_stats = kl_controls[base_name]
                                if gamma == 0.0:
                                    rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                                    used_qvfit = 0.0
                                else:
                                    rep = refit_values_quotient(
                                        base_rep,
                                        keys,
                                        values,
                                        train_qbank,
                                        beta_cfg,
                                        row_scale_power=gamma,
                                    )
                                    used_qvfit = 1.0
                                q_metric_strength = gamma
                                controller_branch = "neutral_fallback" if gamma == 0.0 else "middle_kl"
                                selected_stats = control_stats
                            elif method.endswith("+qfit_diag"):
                                gamma, control_stats, controller_branch = diag_controls[base_name]
                                if gamma == 0.0:
                                    rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                                    used_qvfit = 0.0
                                else:
                                    rep = refit_values_quotient(
                                        base_rep,
                                        keys,
                                        values,
                                        train_qbank,
                                        beta_cfg,
                                        row_scale_power=gamma,
                                    )
                                    used_qvfit = 1.0
                                q_metric_strength = gamma
                                selected_stats = control_stats
                            else:
                                raise ValueError(f"Unsupported method: {method}")

                            qstats = quotient_stats(rep, head_state, holdout_qbank)
                            rows.append(
                                PolicyRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    method=method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
                                    train_zhat_over_zref_cv=stats["zhat_over_zref_cv"],
                                    train_q_row_energy_top5_share=stats["q_row_energy_top5_share"],
                                    selected_q_weight_neff_fraction=selected_stats["q_weight_neff_fraction"],
                                    selected_q_weight_kl_to_neutral=selected_stats["q_weight_kl_to_neutral"],
                                    q_metric_strength=q_metric_strength,
                                    controller_branch=controller_branch,
                                    used_qvfit=used_qvfit,
                                    holdout_l_true=compute_l_true(rep, head_state, holdout_qbank),
                                    holdout_qr_per_dim=qstats["qr_per_dim"],
                                    holdout_qr_cancellation_gain=qstats["qr_cancellation_gain"],
                                    holdout_qr_worst_ratio=qstats["qr_worst_ratio"],
                                )
                            )

    summary = summarize_rows(rows)
    for key, stats in summary.items():
        print(f"\n{key}")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.4f}")

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
    print(f"\nSaved Phase 3C qvfit policy report to {out}")


if __name__ == "__main__":
    main()
