#!/usr/bin/env python3
"""
Phase 3B precursor: candidate-geometry diagnostics without merging.

Keep the support as original tokens only, then ask whether a local
query-conditioned role-compatibility score surfaces better neighbor candidates
than plain adjacency.
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
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import HeadState

from compare_collection_modes import collect_mode_state
from run_experiment import load_model, load_prompt_segments_from_file


LIVE_HYBRID_CFG = HybridSelectorConfig(
    use_delta_b=True,
    use_delta_q_coh=True,
    use_delta_q_span=False,
)


@dataclass
class CandidateGeometryRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    anchor_index: int
    num_window_candidates: int
    best_adjacent_score: float
    best_adjacent_loss: float
    top_compat_index: int
    top_compat_distance: int
    top_compat_score: float
    top_compat_loss: float
    best_utility_index: int
    best_utility_distance: int
    best_utility_loss: float
    mean_local_loss: float
    random_local_loss: float
    top_compat_beats_adjacent: bool
    top_compat_beats_random_local: bool
    top_compat_is_nonadjacent: bool
    any_nonadjacent_outscores_adjacent: bool
    eligible_fraction_vs_adjacent: float
    compat_loss_corr: float


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
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--distance-penalty", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3b_candidate_geometry_q256_t32.json")
    return p.parse_args()


def _build_weighted_design(head_state: HeadState, qbank: QueryBank) -> torch.Tensor:
    queries, weights = qbank.get_weighted_bank()
    queries_f = queries.float()
    keys_f = head_state.keys.float()
    weights_f = weights.float()
    inv_sqrt_d = 1.0 / math.sqrt(max(int(queries_f.shape[1]), 1))
    logits = (queries_f @ keys_f.T) * inv_sqrt_d
    reference_max = logits.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(logits - reference_max)
    row_weights = torch.sqrt(torch.clamp_min(weights_f, 0.0))
    return exp_scores * row_weights.unsqueeze(1)


def _fit_pair_loss(
    *,
    query_tensor: torch.Tensor,
    entry_weights: torch.Tensor,
    left_key: torch.Tensor,
    right_key: torch.Tensor,
    left_value: torch.Tensor,
    right_value: torch.Tensor,
) -> float:
    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    left_logits = (query_tensor @ left_key.unsqueeze(1) * inv_sqrt_d)
    right_logits = (query_tensor @ right_key.unsqueeze(1) * inv_sqrt_d)
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

    key_candidates = [left_key, right_key, 0.5 * (left_key + right_key)]
    value_candidates = [left_value, right_value, 0.5 * (left_value + right_value)]
    best_loss = float("inf")
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
            best_loss = min(best_loss, z_loss + n_loss)
    return best_loss


def _support_indices(rep, head_state: HeadState, atol: float = 1e-6) -> List[int]:
    original_keys = head_state.keys.float()
    support_keys = rep.support_keys.float()
    indices = []
    for sk in support_keys:
        diff = (original_keys - sk.unsqueeze(0)).abs().amax(dim=1)
        indices.append(int(torch.argmin(diff).item()))
    return indices


def summarize(rows: List[dict]) -> dict:
    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["collection_mode"], row["layer"])].append(row)
        grouped[(row["collection_mode"], "all")].append(row)

    for (mode, layer), items in sorted(grouped.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        summary.append(
            {
                "collection_mode": mode,
                "layer": layer,
                "anchors": len(items),
                "mean_top_compat_loss": sum(r["top_compat_loss"] for r in items) / len(items),
                "mean_best_adjacent_loss": sum(r["best_adjacent_loss"] for r in items) / len(items),
                "mean_random_local_loss": sum(r["random_local_loss"] for r in items) / len(items),
                "top_compat_beats_adjacent_frac": sum(r["top_compat_beats_adjacent"] for r in items) / len(items),
                "top_compat_beats_random_local_frac": sum(r["top_compat_beats_random_local"] for r in items) / len(items),
                "top_compat_nonadjacent_frac": sum(r["top_compat_is_nonadjacent"] for r in items) / len(items),
                "any_nonadjacent_outscores_adjacent_frac": sum(r["any_nonadjacent_outscores_adjacent"] for r in items) / len(items),
                "mean_eligible_fraction_vs_adjacent": sum(r["eligible_fraction_vs_adjacent"] for r in items) / len(items),
                "mean_top_compat_distance": sum(r["top_compat_distance"] for r in items) / len(items),
                "mean_best_utility_distance": sum(r["best_utility_distance"] for r in items) / len(items),
                "mean_compat_loss_corr": sum(r["compat_loss_corr"] for r in items) / len(items),
            }
        )
    return {"by_mode_and_layer": summary}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"

    all_rows: List[CandidateGeometryRow] = []
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

                            compat_scores = []
                            fit_losses = []
                            distances = []
                            for j in candidates:
                                role_cos = float(
                                    torch.dot(weighted_design[:, anchor_idx], weighted_design[:, j]).item()
                                    / (column_norm[anchor_idx].item() * column_norm[j].item())
                                )
                                distance = abs(j - anchor_idx)
                                dist_pen = args.distance_penalty * (distance / max(args.window, 1))
                                compat_scores.append(role_cos - dist_pen)
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

                            compat_scores_t = torch.tensor(compat_scores, dtype=torch.float32)
                            fit_losses_t = torch.tensor(fit_losses, dtype=torch.float32)
                            distances_t = torch.tensor(distances, dtype=torch.long)
                            candidate_idx_t = torch.tensor(candidates, dtype=torch.long)

                            top_idx = int(torch.argmax(compat_scores_t).item())
                            best_utility_idx = int(torch.argmin(fit_losses_t).item())

                            adjacent_mask = distances_t == 1
                            if adjacent_mask.any():
                                adjacent_scores = compat_scores_t[adjacent_mask]
                                adjacent_losses = fit_losses_t[adjacent_mask]
                                adjacent_indices = candidate_idx_t[adjacent_mask]
                                best_adj_local = int(torch.argmax(adjacent_scores).item())
                                best_adj_score = float(adjacent_scores[best_adj_local].item())
                                best_adj_loss = float(adjacent_losses[best_adj_local].item())
                                best_adj_index = int(adjacent_indices[best_adj_local].item())
                            else:
                                best_adj_score = float("-inf")
                                best_adj_loss = float("inf")
                                best_adj_index = -1

                            nonadjacent_mask = distances_t > 1
                            any_nonadjacent_outscores_adjacent = bool(
                                nonadjacent_mask.any()
                                and float(compat_scores_t[nonadjacent_mask].max().item()) > best_adj_score
                            )
                            eligible_fraction_vs_adjacent = float(
                                (compat_scores_t > best_adj_score).float().mean().item()
                            ) if math.isfinite(best_adj_score) else 0.0

                            if compat_scores_t.numel() > 1:
                                compat_centered = compat_scores_t - compat_scores_t.mean()
                                loss_centered = fit_losses_t - fit_losses_t.mean()
                                denom = compat_centered.norm().item() * loss_centered.norm().item()
                                compat_loss_corr = float(
                                    torch.dot(compat_centered, loss_centered).item() / max(denom, 1e-12)
                                )
                            else:
                                compat_loss_corr = 0.0

                            random_local_loss = float(fit_losses_t.mean().item())
                            all_rows.append(
                                CandidateGeometryRow(
                                    prompt_id=prompt_name,
                                    collection_mode=collection_mode,
                                    layer=layer_idx,
                                    kv_head=kv_head,
                                    budget=budget,
                                    anchor_index=anchor_idx,
                                    num_window_candidates=len(candidates),
                                    best_adjacent_score=best_adj_score,
                                    best_adjacent_loss=best_adj_loss,
                                    top_compat_index=int(candidate_idx_t[top_idx].item()),
                                    top_compat_distance=int(distances_t[top_idx].item()),
                                    top_compat_score=float(compat_scores_t[top_idx].item()),
                                    top_compat_loss=float(fit_losses_t[top_idx].item()),
                                    best_utility_index=int(candidate_idx_t[best_utility_idx].item()),
                                    best_utility_distance=int(distances_t[best_utility_idx].item()),
                                    best_utility_loss=float(fit_losses_t[best_utility_idx].item()),
                                    mean_local_loss=float(fit_losses_t.mean().item()),
                                    random_local_loss=random_local_loss,
                                    top_compat_beats_adjacent=float(fit_losses_t[top_idx].item()) < best_adj_loss,
                                    top_compat_beats_random_local=float(fit_losses_t[top_idx].item()) < random_local_loss,
                                    top_compat_is_nonadjacent=int(distances_t[top_idx].item()) > 1,
                                    any_nonadjacent_outscores_adjacent=any_nonadjacent_outscores_adjacent,
                                    eligible_fraction_vs_adjacent=eligible_fraction_vs_adjacent,
                                    compat_loss_corr=compat_loss_corr,
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
    print(f"\nSaved candidate-geometry diagnostic to {save_path}", flush=True)


if __name__ == "__main__":
    main()
