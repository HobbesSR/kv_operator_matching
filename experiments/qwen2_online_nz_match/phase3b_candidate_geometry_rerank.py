#!/usr/bin/env python3
"""
Compare the current local compatibility score with one more locality-regularized
rerank before doing any further merge-construction work.
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

from kv_operator_matching.baselines import HybridSelectorConfig, hybrid_support_baseline
from kv_operator_matching.types import HeadState

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file
from phase3b_candidate_geometry import _build_weighted_design, _fit_pair_loss, _support_indices


LIVE_HYBRID_CFG = HybridSelectorConfig(
    use_delta_b=True,
    use_delta_q_coh=True,
    use_delta_q_span=False,
)


@dataclass
class RerankRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    score_name: str
    anchor_index: int
    num_window_candidates: int
    best_adjacent_loss: float
    top_score_distance: int
    top_score_loss: float
    top_score_beats_adjacent: bool
    top_score_beats_random_local: bool
    top_score_is_nonadjacent: bool
    eligible_fraction_vs_adjacent: float


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
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--base-distance-penalty", type=float, default=0.15)
    p.add_argument("--locality-distance-penalty", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3b_candidate_geometry_rerank_q256_t32.json")
    return p.parse_args()


def summarize(rows: List[dict]) -> dict:
    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["score_name"], row["collection_mode"])].append(row)
        grouped[(row["score_name"], "all")].append(row)

    for (score_name, mode), items in sorted(grouped.items()):
        summary.append(
            {
                "score_name": score_name,
                "collection_mode": mode,
                "anchors": len(items),
                "top_score_beats_adjacent_frac": sum(r["top_score_beats_adjacent"] for r in items) / len(items),
                "top_score_beats_random_local_frac": sum(r["top_score_beats_random_local"] for r in items) / len(items),
                "top_score_nonadjacent_frac": sum(r["top_score_is_nonadjacent"] for r in items) / len(items),
                "mean_eligible_fraction_vs_adjacent": sum(r["eligible_fraction_vs_adjacent"] for r in items) / len(items),
                "mean_top_score_distance": sum(r["top_score_distance"] for r in items) / len(items),
                "mean_top_score_loss": sum(r["top_score_loss"] for r in items) / len(items),
                "mean_best_adjacent_loss": sum(r["best_adjacent_loss"] for r in items) / len(items),
            }
        )
    return {"by_score_and_mode": summary}


def _score_current(role_cos: float, distance: int, window: int, base_penalty: float) -> float:
    return role_cos - base_penalty * (distance / max(window, 1))


def _score_locality_regularized(
    role_cos: float,
    distance: int,
    window: int,
    locality_penalty: float,
) -> float:
    if window <= 1:
        return role_cos
    # Keep adjacency as the unpenalized local prior, then penalize farther hops.
    return role_cos - locality_penalty * ((distance - 1) / max(window - 1, 1))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"

    all_rows: List[RerankRow] = []
    skipped = []

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
                kv_states, query_banks, _collection_meta = collect_mode_state(
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

            for layer_idx in args.layers:
                seq_len = kv_states[layer_idx]["keys"].shape[1]
                for kv_head in range(model.config.num_key_value_heads):
                    qbank = query_banks[(layer_idx, kv_head)]
                    train_qbank, _holdout_qbank = qbank.split_train_holdout(args.train_fraction)
                    head_state = HeadState(
                        head_idx=kv_head,
                        layer_idx=layer_idx,
                        keys=kv_states[layer_idx]["keys"][kv_head],
                        values=kv_states[layer_idx]["values"][kv_head],
                    )
                    weighted_design = _build_weighted_design(head_state, train_qbank)
                    column_norm = weighted_design.norm(dim=0).clamp_min(1e-12)
                    queries, weights = train_qbank.get_weighted_bank()
                    queries_f = queries.float()
                    weights_f = weights.float()

                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        rep = hybrid_support_baseline(
                            head_state,
                            train_qbank,
                            budget,
                            config=LIVE_HYBRID_CFG,
                        )
                        anchor_indices = _support_indices(rep, head_state)
                        for anchor_idx in anchor_indices:
                            left = max(0, anchor_idx - args.window)
                            right = min(seq_len, anchor_idx + args.window + 1)
                            candidates = [j for j in range(left, right) if j != anchor_idx]
                            if not candidates:
                                continue

                            role_cos_values = []
                            fit_losses = []
                            distances = []
                            for j in candidates:
                                role_cos = float(
                                    torch.dot(weighted_design[:, anchor_idx], weighted_design[:, j]).item()
                                    / (column_norm[anchor_idx].item() * column_norm[j].item())
                                )
                                distance = abs(j - anchor_idx)
                                role_cos_values.append(role_cos)
                                fit_losses.append(
                                    _fit_pair_loss(
                                        query_tensor=queries_f,
                                        entry_weights=weights_f,
                                        left_key=head_state.keys[anchor_idx].float(),
                                        right_key=head_state.keys[j].float(),
                                        left_value=head_state.values[anchor_idx].float(),
                                        right_value=head_state.values[j].float(),
                                    )
                                )
                                distances.append(distance)

                            role_cos_t = torch.tensor(role_cos_values, dtype=torch.float32)
                            fit_losses_t = torch.tensor(fit_losses, dtype=torch.float32)
                            distances_t = torch.tensor(distances, dtype=torch.long)

                            adjacent_mask = distances_t == 1
                            if not adjacent_mask.any():
                                continue
                            best_adjacent_loss = float(fit_losses_t[adjacent_mask].min().item())

                            score_variants = {
                                "current": torch.tensor(
                                    [
                                        _score_current(
                                            float(role_cos_values[idx]),
                                            int(distances[idx]),
                                            args.window,
                                            args.base_distance_penalty,
                                        )
                                        for idx in range(len(candidates))
                                    ],
                                    dtype=torch.float32,
                                ),
                                "locality_regularized": torch.tensor(
                                    [
                                        _score_locality_regularized(
                                            float(role_cos_values[idx]),
                                            int(distances[idx]),
                                            args.window,
                                            args.locality_distance_penalty,
                                        )
                                        for idx in range(len(candidates))
                                    ],
                                    dtype=torch.float32,
                                ),
                            }

                            for score_name, scores_t in score_variants.items():
                                top_idx = int(torch.argmax(scores_t).item())
                                top_loss = float(fit_losses_t[top_idx].item())
                                top_distance = int(distances_t[top_idx].item())
                                random_local_loss = float(fit_losses_t.mean().item())
                                best_adjacent_score = float(scores_t[adjacent_mask].max().item())
                                nonadjacent_mask = distances_t > 1
                                eligible_fraction_vs_adjacent = float(
                                    (scores_t > best_adjacent_score).float().mean().item()
                                )
                                all_rows.append(
                                    RerankRow(
                                        prompt_id=prompt_name,
                                        collection_mode=collection_mode,
                                        layer=layer_idx,
                                        kv_head=kv_head,
                                        budget=budget,
                                        score_name=score_name,
                                        anchor_index=anchor_idx,
                                        num_window_candidates=len(candidates),
                                        best_adjacent_loss=best_adjacent_loss,
                                        top_score_distance=top_distance,
                                        top_score_loss=top_loss,
                                        top_score_beats_adjacent=top_loss < best_adjacent_loss,
                                        top_score_beats_random_local=top_loss < random_local_loss,
                                        top_score_is_nonadjacent=top_distance > 1,
                                        eligible_fraction_vs_adjacent=eligible_fraction_vs_adjacent,
                                    )
                                )

    payload = {
        "args": vars(args),
        "skipped": skipped,
        "rows": [asdict(row) for row in all_rows],
        "summary": summarize([asdict(row) for row in all_rows]),
    }
    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved candidate rerank diagnostic to {save_path}", flush=True)


if __name__ == "__main__":
    main()
