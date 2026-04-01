#!/usr/bin/env python3
"""
Phase 3C shortlist-policy sweep with fixed downstream OMP + vfit.

This tranche asks whether quotient-aware information is operationally useful as
shortlist construction for a stronger downstream solver, rather than as a
standalone selector.
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

from kv_operator_matching.baselines import (
    compute_attention_mass_scores,
    compute_quotient_omission_scores,
    compute_value_deviation_scores,
    omp_over_shortlist,
    shortlist_indices_from_scores,
)
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_quotient_residual_diagnostics, compute_response
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file


@dataclass
class ShortlistRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    shortlist_policy: str
    shortlist_multiplier: float
    shortlist_size: int
    train_queries: int
    holdout_queries: int
    holdout_l_true: float
    holdout_qr_per_dim: float
    holdout_qr_cancellation_gain: float
    holdout_qr_worst_ratio: float
    shortlist_conformist_fraction: float
    shortlist_conformist_recall: float
    shortlist_mean_mass_score: float
    shortlist_mean_value_deviation: float
    oracle_support_overlap: float
    oracle_support_in_shortlist: float


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
        "--shortlist-policies",
        nargs="+",
        default=["attn_mass", "quotient_omit", "rank_blend", "two_stage_gate"],
    )
    p.add_argument(
        "--shortlist-multipliers",
        nargs="+",
        type=float,
        default=[1.5, 2.0, 3.0, 4.0],
    )
    p.add_argument(
        "--quotient-score-mode",
        choices=["exact_local", "proxy"],
        default="exact_local",
    )
    p.add_argument("--gate-expansion", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3c_shortlist_sweep.json")
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


def match_support_indices(full_keys: torch.Tensor, support_keys: torch.Tensor) -> torch.Tensor:
    remaining = torch.ones(full_keys.shape[0], dtype=torch.bool, device=full_keys.device)
    matched: List[int] = []
    for support_key in support_keys:
        diff = (full_keys - support_key.unsqueeze(0)).abs().amax(dim=1)
        masked = diff.masked_fill(~remaining, float("inf"))
        index = int(torch.argmin(masked).item())
        matched.append(index)
        remaining[index] = False
    return torch.tensor(matched, dtype=torch.long, device=full_keys.device)


def summarize_rows(rows: List[ShortlistRow]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, float, str], list[ShortlistRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.collection_mode, row.shortlist_multiplier, row.shortlist_policy)].append(row)

    baseline_index = {
        (row.prompt_id, row.collection_mode, row.layer, row.kv_head, row.budget, row.shortlist_multiplier): row
        for row in rows
        if row.shortlist_policy == "attn_mass"
    }

    summary: dict[str, dict[str, float]] = {}
    for key, group in sorted(grouped.items()):
        deltas = []
        for row in group:
            baseline = baseline_index.get(
                (row.prompt_id, row.collection_mode, row.layer, row.kv_head, row.budget, row.shortlist_multiplier)
            )
            if baseline is not None:
                deltas.append(row.holdout_l_true - baseline.holdout_l_true)
        summary[f"{key[0]}::m{key[1]:.1f}::{key[2]}"] = {
            "cells": float(len(group)),
            "mean_holdout_l_true": sum(r.holdout_l_true for r in group) / len(group),
            "mean_delta_vs_attn_mass": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "mean_holdout_qr_cancellation_gain": (
                sum(r.holdout_qr_cancellation_gain for r in group) / len(group)
            ),
            "mean_oracle_support_overlap": sum(r.oracle_support_overlap for r in group) / len(group),
            "mean_oracle_support_in_shortlist": (
                sum(r.oracle_support_in_shortlist for r in group) / len(group)
            ),
            "mean_shortlist_conformist_fraction": (
                sum(r.shortlist_conformist_fraction for r in group) / len(group)
            ),
            "mean_shortlist_conformist_recall": (
                sum(r.shortlist_conformist_recall for r in group) / len(group)
            ),
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
    n_kv_heads = model.config.num_key_value_heads
    exact_local = args.quotient_score_mode == "exact_local"
    rows: List[ShortlistRow] = []

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

                    attn_scores = compute_attention_mass_scores(head_state, train_qbank)
                    quotient_scores = compute_quotient_omission_scores(
                        head_state,
                        train_qbank,
                        exact_local=exact_local,
                    )
                    value_dev_scores = compute_value_deviation_scores(head_state, train_qbank)
                    mass_q75 = torch.quantile(attn_scores.float(), 0.75)
                    dev_q25 = torch.quantile(value_dev_scores.float(), 0.25)
                    conformist_mask = (attn_scores >= mass_q75) & (value_dev_scores <= dev_q25)
                    n_conformists = int(conformist_mask.sum().item())

                    cell_records: Dict[Tuple[int, float, str], dict] = {}
                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        for shortlist_multiplier in args.shortlist_multipliers:
                            shortlist_size = min(
                                seq_len,
                                max(budget, int(round(shortlist_multiplier * budget))),
                            )
                            for policy in args.shortlist_policies:
                                shortlist_indices = shortlist_indices_from_scores(
                                    attn_scores,
                                    quotient_scores,
                                    shortlist_size,
                                    policy=policy,
                                    gate_expansion=args.gate_expansion,
                                )
                                base_rep = omp_over_shortlist(
                                    head_state,
                                    train_qbank,
                                    shortlist_indices,
                                    budget,
                                )
                                rep = refit_values(base_rep, keys, values, train_qbank, beta_cfg)
                                support_indices = match_support_indices(keys, rep.support_keys)
                                qstats = quotient_stats(rep, head_state, holdout_qbank)
                                shortlist_conformists = int(conformist_mask[shortlist_indices].sum().item())
                                cell_records[(budget, shortlist_multiplier, policy)] = {
                                    "rep": rep,
                                    "support_indices": support_indices,
                                    "shortlist_indices": shortlist_indices,
                                    "holdout_l_true": compute_l_true(rep, head_state, holdout_qbank),
                                    "qstats": qstats,
                                    "shortlist_conformists": shortlist_conformists,
                                    "shortlist_mean_mass_score": float(attn_scores[shortlist_indices].mean().item()),
                                    "shortlist_mean_value_deviation": float(
                                        value_dev_scores[shortlist_indices].mean().item()
                                    ),
                                }

                    oracle_for_cell: Dict[Tuple[int, float], torch.Tensor] = {}
                    for budget, shortlist_multiplier, _policy in cell_records:
                        key = (budget, shortlist_multiplier)
                        if key not in oracle_for_cell:
                            oracle_policy = min(
                                (
                                    p for (b, m, p) in cell_records
                                    if b == budget and m == shortlist_multiplier
                                ),
                                key=lambda p: cell_records[(budget, shortlist_multiplier, p)]["holdout_l_true"],
                            )
                            oracle_for_cell[key] = cell_records[(budget, shortlist_multiplier, oracle_policy)]["support_indices"]

                    for (budget, shortlist_multiplier, policy), record in cell_records.items():
                        oracle_support = oracle_for_cell[(budget, shortlist_multiplier)]
                        support_indices = record["support_indices"]
                        shortlist_indices = record["shortlist_indices"]
                        oracle_support_set = set(int(x) for x in oracle_support.tolist())
                        support_set = set(int(x) for x in support_indices.tolist())
                        shortlist_set = set(int(x) for x in shortlist_indices.tolist())
                        oracle_overlap = len(oracle_support_set & support_set) / max(len(oracle_support_set), 1)
                        oracle_in_shortlist = len(oracle_support_set & shortlist_set) / max(len(oracle_support_set), 1)
                        rows.append(
                            ShortlistRow(
                                prompt_id=prompt_path.name,
                                collection_mode=collection_mode,
                                layer=layer_idx,
                                kv_head=kv_head,
                                budget=budget,
                                shortlist_policy=policy,
                                shortlist_multiplier=shortlist_multiplier,
                                shortlist_size=int(shortlist_indices.numel()),
                                train_queries=len(train_qbank),
                                holdout_queries=len(holdout_qbank),
                                holdout_l_true=record["holdout_l_true"],
                                holdout_qr_per_dim=record["qstats"]["qr_per_dim"],
                                holdout_qr_cancellation_gain=record["qstats"]["qr_cancellation_gain"],
                                holdout_qr_worst_ratio=record["qstats"]["qr_worst_ratio"],
                                shortlist_conformist_fraction=(
                                    record["shortlist_conformists"] / max(int(shortlist_indices.numel()), 1)
                                ),
                                shortlist_conformist_recall=(
                                    record["shortlist_conformists"] / max(n_conformists, 1)
                                ),
                                shortlist_mean_mass_score=record["shortlist_mean_mass_score"],
                                shortlist_mean_value_deviation=record["shortlist_mean_value_deviation"],
                                oracle_support_overlap=float(oracle_overlap),
                                oracle_support_in_shortlist=float(oracle_in_shortlist),
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
    print(f"\nSaved Phase 3C shortlist sweep report to {out}")


if __name__ == "__main__":
    main()
