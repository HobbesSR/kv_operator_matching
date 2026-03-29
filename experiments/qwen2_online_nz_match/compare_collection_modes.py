#!/usr/bin/env python3
"""
Small Phase 2 comparison of query collection regimes.

Loads the model once, then compares `online` versus `repeat-prefill`
reference-query collection across a small prompt/layer/budget matrix using the
same downstream baseline/refit machinery.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.config import BetaFitConfig, QueryBankConfig
from kv_operator_matching.baselines import attention_mass_baseline
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import HeadState

from run_experiment import (
    build_repeat_prefill_prompt,
    collect_online_kv_and_query_banks,
    collect_teacher_forced_kv_and_query_banks,
    build_query_bank,
    evaluate_good_support,
    extract_teacher_forced_continuation_ids,
    extract_kv_states_from_prompt,
    extract_query_states_from_prompt,
    load_prompt_segments_from_file,
    load_model,
    run_methods,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", nargs="+", type=int, default=[4, 20])
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
            "near_capacity_dispatch_safe.json",
            "network_cutover.json",
            "relational_binding_probe.json",
        ],
    )
    p.add_argument("--prefix-turns", type=int, default=8)
    p.add_argument("--continuation-turns", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/collection_mode_comparison.json")
    return p.parse_args()

def collect_mode_state(
    *,
    mode: str,
    prompt: str,
    continuation_text: str,
    model,
    tokenizer,
    device: str,
    layers: List[int],
    max_queries: int,
    max_new_tokens: int,
    prefill_chunk_size: int,
) -> Tuple[Dict, Dict[Tuple[int, int], QueryBank], dict]:
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    qpk = n_heads // n_kv_heads

    if mode == "online":
        bank_cfg = QueryBankConfig(max_queries=max_queries, weighting_scheme="recency")
        kv_states, query_banks, _generated_tokens = collect_online_kv_and_query_banks(
            model,
            tokenizer,
            prompt,
            device,
            layers,
            bank_cfg,
            max_new_tokens=max_new_tokens,
            prefill_chunk_size=prefill_chunk_size,
        )
        observed_positions = _generated_tokens
        retained_queries = len(next(iter(query_banks.values())))
        return kv_states, query_banks, {
            "collection_mode": mode,
            "observed_positions": observed_positions,
            "query_heads_per_kv_head": qpk,
            "raw_query_vectors_per_bank": observed_positions * qpk,
            "retained_query_vectors_per_bank": retained_queries,
            "retained_position_equivalent": retained_queries / qpk,
            "retained_opportunity_fraction": (
                (retained_queries / qpk) / max(observed_positions, 1)
            ),
        }

    if mode == "teacher-forced":
        bank_cfg = QueryBankConfig(max_queries=max_queries, weighting_scheme="uniform")
        continuation_token_ids = extract_teacher_forced_continuation_ids(
            tokenizer,
            prompt,
            continuation_text,
        )
        kv_states, query_banks, observed_tokens = collect_teacher_forced_kv_and_query_banks(
            model,
            tokenizer,
            prompt,
            continuation_token_ids,
            device,
            layers,
            bank_cfg,
            prefill_chunk_size=prefill_chunk_size,
            max_continuation_tokens=max_new_tokens,
        )
        retained_queries = len(next(iter(query_banks.values())))
        return kv_states, query_banks, {
            "collection_mode": mode,
            "observed_positions": observed_tokens,
            "query_heads_per_kv_head": qpk,
            "raw_query_vectors_per_bank": observed_tokens * qpk,
            "retained_query_vectors_per_bank": retained_queries,
            "retained_position_equivalent": retained_queries / qpk,
            "retained_opportunity_fraction": (
                (retained_queries / qpk) / max(observed_tokens, 1)
            ),
        }

    if mode != "repeat-prefill":
        raise ValueError(f"Unsupported collection mode: {mode}")

    bank_cfg = QueryBankConfig(max_queries=max_queries, weighting_scheme="uniform")
    kv_states = extract_kv_states_from_prompt(model, tokenizer, prompt, device, layers)
    query_states = extract_query_states_from_prompt(
        model,
        tokenizer,
        build_repeat_prefill_prompt(prompt),
        device,
        layers,
    )
    query_banks: Dict[Tuple[int, int], QueryBank] = {}
    for layer_idx in layers:
        for kv_head in range(n_kv_heads):
            query_banks[(layer_idx, kv_head)] = build_query_bank(
                query_states[layer_idx],
                kv_head,
                n_heads,
                n_kv_heads,
                bank_cfg,
            )
    observed_positions = query_states[layers[0]].shape[1]
    retained_queries = len(next(iter(query_banks.values())))
    return kv_states, query_banks, {
        "collection_mode": mode,
        "observed_positions": observed_positions,
        "query_heads_per_kv_head": qpk,
        "raw_query_vectors_per_bank": observed_positions * qpk,
        "retained_query_vectors_per_bank": retained_queries,
        "retained_position_equivalent": retained_queries / qpk,
        "retained_opportunity_fraction": (
            (retained_queries / qpk) / max(observed_positions, 1)
        ),
    }


def summarize_holdout_within_mode(rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] != "holdout":
            continue
        grouped[(row["collection_mode"], row["method"])].append(row["l_true"])

    summary = []
    for (collection_mode, method), values in sorted(grouped.items()):
        summary.append(
            {
                "collection_mode": collection_mode,
                "method": method,
                "mean_holdout_l_true_within_mode": sum(values) / len(values),
                "num_rows": len(values),
            }
        )
    return summary


def summarize_deltas(rows: List[dict]) -> List[dict]:
    baseline_for = {
        "recency+refit": "recency",
        "recency+vfit": "recency",
        "recency+phase1b": "recency",
        "attn_mass+refit": "attn_mass",
        "attn_mass+vfit": "attn_mass",
        "attn_mass+phase1b": "attn_mass",
    }

    by_cell = {}
    for row in rows:
        if row["split"] != "holdout":
            continue
        key = (
            row["prompt_id"],
            row["collection_mode"],
            row["layer"],
            row["kv_head"],
            row["budget"],
            row["method"],
        )
        by_cell[key] = row["l_true"]

    summary = []
    for method, baseline in baseline_for.items():
        deltas_by_mode = defaultdict(list)
        for key, l_true in by_cell.items():
            prompt_id, collection_mode, layer, kv_head, budget, method_name = key
            if method_name != method:
                continue
            baseline_key = (prompt_id, collection_mode, layer, kv_head, budget, baseline)
            if baseline_key not in by_cell:
                continue
            deltas_by_mode[collection_mode].append(l_true - by_cell[baseline_key])

        for collection_mode, deltas in sorted(deltas_by_mode.items()):
            summary.append(
                {
                    "collection_mode": collection_mode,
                    "method": method,
                    "baseline": baseline,
                    "mean_delta_holdout_l_true": sum(deltas) / len(deltas),
                    "improved_cells": sum(delta < 0 for delta in deltas),
                    "num_cells": len(deltas),
                }
            )
    return summary


def summarize_paired_delta_differences(rows: List[dict]) -> List[dict]:
    """
    Compare collection modes only through baseline-relative deltas.

    Raw holdout losses are not directly comparable across collection modes
    because the query-bank distributions differ. The paired quantity we trust is:

      (method - baseline) under mode_a
        minus
      (method - baseline) under mode_b
    """
    baseline_for = {
        "recency+refit": "recency",
        "recency+vfit": "recency",
        "recency+phase1b": "recency",
        "attn_mass+refit": "attn_mass",
        "attn_mass+vfit": "attn_mass",
        "attn_mass+phase1b": "attn_mass",
    }

    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    by_mode = {}
    for row in holdout_rows:
        key = (
            row["prompt_id"],
            row["collection_mode"],
            row["layer"],
            row["kv_head"],
            row["budget"],
            row["method"],
        )
        by_mode[key] = row["l_true"]

    mode_pairs = [
        ("online", "teacher-forced"),
        ("teacher-forced", "repeat-prefill"),
        ("online", "repeat-prefill"),
    ]
    summary = []
    for method, baseline in baseline_for.items():
        for mode_a, mode_b in mode_pairs:
            paired_diffs = []
            for row in holdout_rows:
                if row["method"] != method or row["collection_mode"] != mode_a:
                    continue
                base_key_a = (
                    row["prompt_id"],
                    mode_a,
                    row["layer"],
                    row["kv_head"],
                    row["budget"],
                    baseline,
                )
                method_key_b = (
                    row["prompt_id"],
                    mode_b,
                    row["layer"],
                    row["kv_head"],
                    row["budget"],
                    method,
                )
                base_key_b = (
                    row["prompt_id"],
                    mode_b,
                    row["layer"],
                    row["kv_head"],
                    row["budget"],
                    baseline,
                )
                if (
                    base_key_a not in by_mode
                    or method_key_b not in by_mode
                    or base_key_b not in by_mode
                ):
                    continue
                delta_a = row["l_true"] - by_mode[base_key_a]
                delta_b = by_mode[method_key_b] - by_mode[base_key_b]
                paired_diffs.append(delta_a - delta_b)

            if paired_diffs:
                summary.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        "mean_delta_difference": sum(paired_diffs) / len(paired_diffs),
                        f"{mode_a}_better_cells": sum(diff < 0 for diff in paired_diffs),
                        f"{mode_b}_better_cells": sum(diff > 0 for diff in paired_diffs),
                        "num_cells": len(paired_diffs),
                    }
                )
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model, args.device)
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    print(f"  Heads: {n_heads} query / {n_kv_heads} KV", flush=True)

    beta_cfg = BetaFitConfig(
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
    )

    prompt_base = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    prompt_specs = [(name, prompt_base / name) for name in args.prompt_files]

    all_rows: List[dict] = []
    all_good_support: List[dict] = []
    opportunity_stats: List[dict] = []
    skipped: List[dict] = []

    for prompt_name, prompt_path in prompt_specs:
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
                print(
                    f"  skipped prompt={prompt_name} mode={collection_mode}: {exc}",
                    flush=True,
                )
                skipped.append(
                    {
                        "prompt_id": prompt_name,
                        "collection_mode": collection_mode,
                        "reason": str(exc),
                    }
                )
                continue
            opportunity_stats.append(
                {
                    "prompt_id": prompt_name,
                    **collection_meta,
                }
            )

            for layer_idx in args.layers:
                seq_len = kv_states[layer_idx]["keys"].shape[1]
                for kv_head in range(n_kv_heads):
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
                        results, _diagnostics = run_methods(
                            head_state,
                            train_qbank,
                            holdout_qbank,
                            budget,
                            beta_cfg,
                            args.seed,
                            "full",
                        )
                        for row in results:
                            payload = asdict(row)
                            payload["prompt_id"] = prompt_name
                            payload["collection_mode"] = collection_mode
                            all_rows.append(payload)

                        attn_rows = [row for row in results if row.method == "attn_mass" and row.split == "holdout"]
                        if attn_rows:
                            support_diag = evaluate_good_support(
                                rep=attention_mass_baseline(head_state, train_qbank, budget),
                                head_state=head_state,
                                train_qbank=train_qbank,
                                holdout_qbank=holdout_qbank,
                                layer=layer_idx,
                                kv_head=kv_head,
                                budget=budget,
                                args=argparse.Namespace(
                                    good_support_ltrue_threshold=10.0,
                                    good_support_stable_rank_threshold=4.0,
                                    good_support_cond_threshold=1e8,
                                    good_support_top5_share_threshold=0.5,
                                ),
                            )
                            diag_payload = asdict(support_diag)
                            diag_payload["prompt_id"] = prompt_name
                            diag_payload["collection_mode"] = collection_mode
                            all_good_support.append(diag_payload)

    summary = {
        "rows": all_rows,
        "good_support": all_good_support,
        "opportunity_stats": opportunity_stats,
        "skipped": skipped,
        "mean_holdout_l_true_within_mode": summarize_holdout_within_mode(all_rows),
        "delta_vs_baseline": summarize_deltas(all_rows),
        "paired_delta_differences": summarize_paired_delta_differences(all_rows),
    }

    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
