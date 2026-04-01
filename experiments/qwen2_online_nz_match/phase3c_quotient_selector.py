#!/usr/bin/env python3
"""
Phase 3C quotient-aware selector experiment.

This script tests the first direct selector-side use of the quotient-residual
story: retain tokens by bank-aggregated single-atom omission damage rather than
by attention mass alone, then compare against attn-mass, OMP, and hybrid
supports with and without anchored value repair.
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
    attention_mass_baseline,
    hybrid_support_baseline,
    omp_mass_baseline,
    quotient_omit_omp_baseline,
    quotient_omission_baseline,
)
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_quotient_residual_diagnostics, compute_response
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file


@dataclass
class SelectorRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    train_queries: int
    holdout_queries: int
    holdout_l_true: float
    holdout_qr_per_dim: float
    holdout_qr_cancellation_gain: float
    holdout_qr_worst_ratio: float


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", nargs="+", type=int, default=[4, 12, 20, 28])
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
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "attn_mass",
            "attn_mass+vfit",
            "quotient_omit",
            "quotient_omit+vfit",
            "quotient_omit_omp",
            "quotient_omit_omp+vfit",
            "omp",
            "omp+vfit",
            "hybrid",
            "hybrid+vfit",
        ],
    )
    p.add_argument(
        "--quotient-score-mode",
        choices=["exact_local", "proxy"],
        default="exact_local",
    )
    p.add_argument("--shortlist-multiplier", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3c_quotient_selector.json")
    return p.parse_args()


def compute_l_true(rep: CompactRepresentation, head_state: HeadState, qbank: QueryBank) -> float:
    queries, weights = qbank.get_weighted_bank()
    total_w = weights.sum().clamp(min=1e-12)
    l_true = (
        ((compute_response(queries, rep.support_keys, rep.support_values, rep.betas) -
          compute_response(queries, head_state.keys, head_state.values)) ** 2).sum(dim=-1) * weights
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


def summarize_rows(rows: List[SelectorRow]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[SelectorRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.collection_mode, row.method)].append(row)

    baseline_index = {
        (row.prompt_id, row.collection_mode, row.layer, row.kv_head, row.budget): row
        for row in rows
        if row.method == "attn_mass"
    }

    summary: dict[str, dict[str, float]] = {}
    for key, group in sorted(grouped.items()):
        deltas = []
        for row in group:
            baseline = baseline_index.get(
                (row.prompt_id, row.collection_mode, row.layer, row.kv_head, row.budget)
            )
            if baseline is not None:
                deltas.append(row.holdout_l_true - baseline.holdout_l_true)
        summary[f"{key[0]}::{key[1]}"] = {
            "cells": float(len(group)),
            "mean_holdout_l_true": sum(r.holdout_l_true for r in group) / len(group),
            "mean_holdout_qr_per_dim": sum(r.holdout_qr_per_dim for r in group) / len(group),
            "mean_holdout_qr_cancellation_gain": (
                sum(r.holdout_qr_cancellation_gain for r in group) / len(group)
            ),
            "mean_holdout_qr_worst_ratio": sum(r.holdout_qr_worst_ratio for r in group) / len(group),
            "mean_delta_vs_attn_mass": (sum(deltas) / len(deltas)) if deltas else 0.0,
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
    rows: List[SelectorRow] = []
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
                        if any(method.startswith("quotient_omit") for method in args.methods):
                            base_reps["quotient_omit"] = quotient_omission_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                exact_local=exact_local,
                            )
                        if any(method.startswith("quotient_omit_omp") for method in args.methods):
                            base_reps["quotient_omit_omp"] = quotient_omit_omp_baseline(
                                head_state,
                                train_qbank,
                                budget,
                                shortlist_multiplier=args.shortlist_multiplier,
                                exact_local=exact_local,
                            )
                        if any(method.startswith("omp") for method in args.methods):
                            base_reps["omp"] = omp_mass_baseline(head_state, train_qbank, budget)
                        if any(method.startswith("hybrid") for method in args.methods):
                            base_reps["hybrid"] = hybrid_support_baseline(head_state, train_qbank, budget)

                        for method in args.methods:
                            if method.endswith("+vfit"):
                                base_name = method[:-5]
                                rep = refit_values(
                                    base_reps[base_name],
                                    keys,
                                    values,
                                    train_qbank,
                                    beta_cfg,
                                )
                            else:
                                rep = base_reps[method]

                            qstats = quotient_stats(rep, head_state, holdout_qbank)
                            rows.append(
                                SelectorRow(
                                    prompt_id=prompt_path.name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    method=method,
                                    train_queries=len(train_qbank),
                                    holdout_queries=len(holdout_qbank),
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
    print(f"\nSaved Phase 3C quotient selector report to {out}")


if __name__ == "__main__":
    main()
