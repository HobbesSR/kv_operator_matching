#!/usr/bin/env python3
"""
Phase 3B: first constructed-support tranche with adjacent pair-merge candidates.

This keeps the Phase 3A selector core fixed at `ΔB + ΔQ_coh` and tests one
small substrate shift only:

- original-token hybrid selector
- conflict-aware hybrid selector over original tokens plus adjacent pair merges

Each support is evaluated before and after anchored value refit on the same
collection surfaces used in late Phase 3A.
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
    hybrid_pairmerge_support_baseline,
    hybrid_support_baseline,
)
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_quotient_residual_diagnostics
from kv_operator_matching.types import HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from run_experiment import compute_metrics, load_model, load_prompt_segments_from_file


LIVE_HYBRID_CFG = HybridSelectorConfig(
    use_delta_b=True,
    use_delta_q_coh=True,
    use_delta_q_span=False,
)


@dataclass
class Phase3BRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    split: str
    n_tokens: int
    train_queries: int
    holdout_queries: int
    l_z: float
    l_n_per_dim: float
    l_true: float
    l_lin: float
    merged_fraction: float
    qr_per_dim: float
    qr_cancellation_gain: float
    qr_worst_ratio: float


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
    p.add_argument("--continuation-turns", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3b_pairmerge.json")
    return p.parse_args()


def _merged_atom_fraction(rep, head_state: HeadState, atol: float = 1e-6) -> float:
    if rep.support_keys.numel() == 0:
        return 0.0
    original_keys = head_state.keys.float()
    support_keys = rep.support_keys.float()
    diff = (support_keys[:, None, :] - original_keys[None, :, :]).abs().amax(dim=-1)
    matched_original = diff.le(atol).any(dim=1)
    return float((~matched_original).float().mean().item())


def _record_rows(
    rows: List[Phase3BRow],
    *,
    rep,
    method: str,
    prompt_id: str,
    collection_mode: str,
    head_state: HeadState,
    train_qbank,
    holdout_qbank,
    budget: int,
):
    merged_fraction = _merged_atom_fraction(rep, head_state)
    for split_name, bank in (("train", train_qbank), ("holdout", holdout_qbank)):
        lz, ln, lt, llin = compute_metrics(rep, head_state, bank)
        qr = quotient_stats(rep, head_state, bank)
        rows.append(
            Phase3BRow(
                prompt_id=prompt_id,
                collection_mode=collection_mode,
                layer=head_state.layer_idx,
                kv_head=head_state.head_idx,
                budget=budget,
                method=method,
                split=split_name,
                n_tokens=int(head_state.keys.shape[0]),
                train_queries=len(train_qbank),
                holdout_queries=len(holdout_qbank),
                l_z=lz,
                l_n_per_dim=ln,
                l_true=lt,
                l_lin=llin,
                merged_fraction=merged_fraction,
                qr_per_dim=qr["qr_per_dim"],
                qr_cancellation_gain=qr["qr_cancellation_gain"],
                qr_worst_ratio=qr["qr_worst_ratio"],
            )
        )


def quotient_stats(rep, head_state: HeadState, qbank):
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


def summarize(rows: List[dict]) -> dict:
    holdout_rows = [row for row in rows if row["split"] == "holdout"]

    mean_by_mode_method = []
    grouped = defaultdict(list)
    merged_grouped = defaultdict(list)
    for row in holdout_rows:
        grouped[(row["collection_mode"], row["method"])].append(row["l_true"])
        merged_grouped[(row["collection_mode"], row["method"])].append(row["merged_fraction"])
    for (mode, method), values in sorted(grouped.items()):
        mean_by_mode_method.append(
            {
                "collection_mode": mode,
                "method": method,
                "mean_holdout_l_true": sum(values) / len(values),
                "mean_merged_fraction": sum(merged_grouped[(mode, method)]) / len(
                    merged_grouped[(mode, method)]
                ),
                "mean_holdout_qr_per_dim": sum(
                    row["qr_per_dim"] for row in holdout_rows
                    if row["collection_mode"] == mode and row["method"] == method
                ) / len(values),
                "mean_holdout_qr_cancellation_gain": sum(
                    row["qr_cancellation_gain"] for row in holdout_rows
                    if row["collection_mode"] == mode and row["method"] == method
                ) / len(values),
                "num_cells": len(values),
            }
        )

    by_cell = {}
    for row in holdout_rows:
        key = (
            row["prompt_id"],
            row["collection_mode"],
            row["layer"],
            row["kv_head"],
            row["budget"],
            row["method"],
        )
        by_cell[key] = row

    comparisons = []
    for lhs, rhs in [
        ("hybrid_pairmerge", "hybrid"),
        ("hybrid_pairmerge+vfit", "hybrid+vfit"),
    ]:
        deltas_by_mode = defaultdict(list)
        qr_deltas_by_mode = defaultdict(list)
        merged_by_mode = defaultdict(list)
        for key, lhs_row in by_cell.items():
            prompt_id, mode, layer, kv_head, budget, method = key
            if method != lhs:
                continue
            rhs_key = (prompt_id, mode, layer, kv_head, budget, rhs)
            rhs_row = by_cell.get(rhs_key)
            if rhs_row is None:
                continue
            deltas_by_mode[mode].append(lhs_row["l_true"] - rhs_row["l_true"])
            qr_deltas_by_mode[mode].append(lhs_row["qr_per_dim"] - rhs_row["qr_per_dim"])
            merged_by_mode[mode].append(lhs_row["merged_fraction"])

        for mode, values in sorted(deltas_by_mode.items()):
            comparisons.append(
                {
                    "collection_mode": mode,
                    "lhs": lhs,
                    "rhs": rhs,
                    "mean_delta_holdout_l_true": sum(values) / len(values),
                    "mean_delta_holdout_qr_per_dim": sum(qr_deltas_by_mode[mode]) / len(qr_deltas_by_mode[mode]),
                    "improved_cells": sum(value < 0 for value in values),
                    "num_cells": len(values),
                    "mean_lhs_merged_fraction": sum(merged_by_mode[mode]) / len(
                        merged_by_mode[mode]
                    ),
                }
            )

    return {
        "mean_holdout_l_true_by_method": mean_by_mode_method,
        "pairmerge_vs_original": comparisons,
    }


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.device)
    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"

    beta_cfg = BetaFitConfig(
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
        max_fit_queries=args.max_queries,
    )

    rows: List[Phase3BRow] = []
    skipped = []
    opportunity_stats = []

    for prompt_name in args.prompt_files:
        prompt_path = Path(prompt_name)
        if not prompt_path.is_absolute():
            prompt_path = smoke_data_path / prompt_name
        prompt, continuation_text = load_prompt_segments_from_file(
            prompt_path,
            prefix_turns=args.prefix_turns,
            continuation_turns=args.continuation_turns,
        )

        for collection_mode in args.collection_modes:
            print(f"\n=== prompt={prompt_name} mode={collection_mode} ===", flush=True)
            try:
                kv_states, query_banks, collection_meta = collect_mode_state(
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
                skipped.append(
                    {
                        "prompt_id": prompt_name,
                        "collection_mode": collection_mode,
                        "reason": str(exc),
                    }
                )
                print(f"  skipped: {exc}", flush=True)
                continue

            opportunity_stats.append({"prompt_id": prompt_name, **collection_meta})

            for layer_idx in args.layers:
                seq_len = kv_states[layer_idx]["keys"].shape[1]
                for kv_head in range(model.config.num_key_value_heads):
                    qbank = query_banks[(layer_idx, kv_head)]
                    train_qbank, holdout_qbank = qbank.split_train_holdout(args.train_fraction)
                    head_state = HeadState(
                        head_idx=kv_head,
                        layer_idx=layer_idx,
                        keys=kv_states[layer_idx]["keys"][kv_head],
                        values=kv_states[layer_idx]["values"][kv_head],
                    )

                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        beta_cfg.support_size = budget

                        rep_h = hybrid_support_baseline(
                            head_state, train_qbank, budget, config=LIVE_HYBRID_CFG
                        )
                        rep_h_v = refit_values(
                            rep_h,
                            head_state.keys,
                            head_state.values,
                            train_qbank,
                            beta_cfg,
                        )
                        rep_pm = hybrid_pairmerge_support_baseline(
                            head_state, train_qbank, budget, config=LIVE_HYBRID_CFG
                        )
                        rep_pm_v = refit_values(
                            rep_pm,
                            head_state.keys,
                            head_state.values,
                            train_qbank,
                            beta_cfg,
                        )

                        _record_rows(
                            rows,
                            rep=rep_h,
                            method="hybrid",
                            prompt_id=prompt_name,
                            collection_mode=collection_mode,
                            head_state=head_state,
                            train_qbank=train_qbank,
                            holdout_qbank=holdout_qbank,
                            budget=budget,
                        )
                        _record_rows(
                            rows,
                            rep=rep_h_v,
                            method="hybrid+vfit",
                            prompt_id=prompt_name,
                            collection_mode=collection_mode,
                            head_state=head_state,
                            train_qbank=train_qbank,
                            holdout_qbank=holdout_qbank,
                            budget=budget,
                        )
                        _record_rows(
                            rows,
                            rep=rep_pm,
                            method="hybrid_pairmerge",
                            prompt_id=prompt_name,
                            collection_mode=collection_mode,
                            head_state=head_state,
                            train_qbank=train_qbank,
                            holdout_qbank=holdout_qbank,
                            budget=budget,
                        )
                        _record_rows(
                            rows,
                            rep=rep_pm_v,
                            method="hybrid_pairmerge+vfit",
                            prompt_id=prompt_name,
                            collection_mode=collection_mode,
                            head_state=head_state,
                            train_qbank=train_qbank,
                            holdout_qbank=holdout_qbank,
                            budget=budget,
                        )

    payload = {
        "args": vars(args),
        "opportunity_stats": opportunity_stats,
        "skipped": skipped,
        "rows": [asdict(row) for row in rows],
        "summary": summarize([asdict(row) for row in rows]),
    }
    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved Phase 3B comparison to {save_path}", flush=True)


if __name__ == "__main__":
    main()
