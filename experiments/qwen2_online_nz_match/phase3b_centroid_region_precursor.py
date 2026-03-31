#!/usr/bin/env python3
"""
Phase 3B precursor: compare locality-only versus centroid-conditioned regional
assignment around the fixed Phase 3A hybrid anchors.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import (
    HybridSelectorConfig,
    _build_anchor_region_atoms,
    _build_mass_frame,
    _match_support_to_source_indices,
    hybrid_support_baseline,
)
from kv_operator_matching.config import BetaFitConfig
from kv_operator_matching.objectives import compute_logits
from kv_operator_matching.types import HeadState
from kv_operator_matching.value_fit import refit_values

from compare_collection_modes import collect_mode_state
from forensic_support_geometry import build_design, delta_stats, matrix_stats
from run_experiment import compute_metrics, load_model, load_prompt_segments_from_file


LIVE_HYBRID_CFG = HybridSelectorConfig(
    use_delta_b=True,
    use_delta_q_coh=True,
    use_delta_q_span=False,
)


@dataclass
class CentroidRegionRow:
    prompt_id: str
    collection_mode: str
    layer: int
    kv_head: int
    budget: int
    method: str
    split: str
    l_true: float
    l_z: float
    l_n_per_dim: float
    changed_atom_fraction: float
    assigned_token_fraction: float
    mean_region_size: float
    mean_assignment_similarity: float
    design_stable_rank: float
    low_sv_delta_share: float


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
    p.add_argument("--assignment-window", type=int, default=16)
    p.add_argument("--assignment-distance-penalty", type=float, default=0.15)
    p.add_argument("--assignment-score-floor", type=float, default=0.0)
    p.add_argument("--max-neighbor-blend", type=float, default=0.5)
    p.add_argument("--num-centroids", type=int, default=8)
    p.add_argument("--kmeans-iters", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="results/scratch/phase3b_centroid_region_precursor_q256_t32.json")
    return p.parse_args()


def _changed_atom_fraction(rep, head_state: HeadState, atol: float = 1e-6) -> float:
    if rep.support_keys.numel() == 0:
        return 0.0
    original_keys = head_state.keys.float()
    support_keys = rep.support_keys.float()
    diff = (support_keys[:, None, :] - original_keys[None, :, :]).abs().amax(dim=-1)
    matched_original = diff.le(atol).any(dim=1)
    return float((~matched_original).float().mean().item())


def _weighted_kmeans_centroids(
    queries: torch.Tensor,
    weights: torch.Tensor,
    *,
    k: int,
    iters: int,
) -> torch.Tensor:
    n = queries.shape[0]
    if n == 0:
        return queries[:0]
    k = max(1, min(k, n))
    order = torch.argsort(weights, descending=True)
    centroids = queries[order[:k]].clone()
    if k == n:
        return centroids
    for _ in range(iters):
        d2 = torch.cdist(queries.float(), centroids.float()).pow(2)
        assign = d2.argmin(dim=1)
        new_centroids = []
        for idx in range(k):
            mask = assign == idx
            if not mask.any():
                new_centroids.append(centroids[idx : idx + 1])
                continue
            w = weights[mask].float()
            q = queries[mask].float()
            new_centroids.append(((w.unsqueeze(1) * q).sum(dim=0) / w.sum().clamp_min(1e-12)).unsqueeze(0))
        centroids = torch.cat(new_centroids, dim=0).to(dtype=queries.dtype, device=queries.device)
    return centroids


def _centroid_conditioned_region_atoms(
    *,
    keys_f: torch.Tensor,
    values_f: torch.Tensor,
    anchor_indices: list[int],
    centroids: torch.Tensor,
    assignment_window: int,
    assignment_distance_penalty: float,
    assignment_score_floor: float,
    max_neighbor_blend: float,
) -> dict[str, torch.Tensor]:
    if not anchor_indices:
        empty_long = torch.empty(0, dtype=torch.long, device=keys_f.device)
        empty_float = torch.empty(0, dtype=torch.float32, device=keys_f.device)
        return {
            "support_keys": keys_f[:0],
            "support_values": values_f[:0],
            "assignments": empty_long,
            "assignment_scores": empty_float,
            "region_sizes": empty_long,
        }

    centroid_logits = compute_logits(centroids.float(), keys_f.float())
    ref_max = centroid_logits.max(dim=1, keepdim=True).values
    fingerprint = torch.exp(centroid_logits - ref_max).T  # (n_tokens, n_centroids)
    fingerprint = torch.nn.functional.normalize(fingerprint, dim=1)

    device = keys_f.device
    anchor_index_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
    anchor_fp = fingerprint[anchor_index_t]
    similarity = fingerprint @ anchor_fp.T

    token_indices = torch.arange(keys_f.shape[0], device=device, dtype=torch.long)
    distances = (token_indices.unsqueeze(1) - anchor_index_t.unsqueeze(0)).abs()
    if assignment_window <= 0:
        local_mask = distances == 0
        distance_penalty = torch.zeros_like(similarity)
    else:
        local_mask = distances <= assignment_window
        denom = max(assignment_window - 1, 1)
        distance_penalty = assignment_distance_penalty * (
            distances.clamp_min(1).float() - 1.0
        ) / denom

    score = similarity - distance_penalty
    score = score.masked_fill(~local_mask, -float("inf"))
    for anchor_pos, anchor_idx in enumerate(anchor_indices):
        score[anchor_idx, anchor_pos] = float("inf")

    best_score, best_anchor_pos = torch.max(score, dim=1)
    assigned_mask = best_score > assignment_score_floor
    assigned_mask[anchor_index_t] = True
    assignments = torch.full((keys_f.shape[0],), -1, dtype=torch.long, device=device)
    assignments[assigned_mask] = best_anchor_pos[assigned_mask]

    assignment_scores = torch.zeros(keys_f.shape[0], dtype=torch.float32, device=device)
    assignment_scores[assigned_mask] = torch.where(
        torch.isfinite(best_score[assigned_mask]),
        best_score[assigned_mask].to(dtype=torch.float32),
        torch.ones_like(best_score[assigned_mask], dtype=torch.float32),
    )

    support_keys = []
    support_values = []
    region_sizes = []
    for anchor_pos, anchor_idx in enumerate(anchor_indices):
        member_mask = assignments == anchor_pos
        members = torch.nonzero(member_mask, as_tuple=False).squeeze(1)
        if members.numel() == 0:
            members = torch.tensor([anchor_idx], dtype=torch.long, device=device)
        region_sizes.append(int(members.numel()))

        member_scores = assignment_scores[members].clamp_min(0.0)
        anchor_member = members == anchor_idx
        if anchor_member.any():
            member_scores[anchor_member] = 1.0
        else:
            members = torch.cat([torch.tensor([anchor_idx], dtype=torch.long, device=device), members], dim=0)
            member_scores = torch.cat([torch.ones(1, dtype=torch.float32, device=device), member_scores], dim=0)

        weight_sum = member_scores.sum().clamp_min(1e-12)
        mean_key = (member_scores.unsqueeze(1) * keys_f[members]).sum(dim=0) / weight_sum
        mean_value = (member_scores.unsqueeze(1) * values_f[members]).sum(dim=0) / weight_sum

        neighbor_weight = member_scores[(members != anchor_idx)].sum()
        blend = float(
            min(
                max_neighbor_blend,
                float((neighbor_weight / (1.0 + neighbor_weight)).item()) if neighbor_weight.numel() > 0 else 0.0,
            )
        )
        anchor_key = keys_f[anchor_idx]
        anchor_value = values_f[anchor_idx]
        support_keys.append(((1.0 - blend) * anchor_key + blend * mean_key).unsqueeze(0))
        support_values.append(((1.0 - blend) * anchor_value + blend * mean_value).unsqueeze(0))

    return {
        "support_keys": torch.cat(support_keys, dim=0).to(dtype=keys_f.dtype, device=device),
        "support_values": torch.cat(support_values, dim=0).to(dtype=values_f.dtype, device=values_f.device),
        "assignments": assignments,
        "assignment_scores": assignment_scores,
        "region_sizes": torch.tensor(region_sizes, dtype=torch.long, device=device),
    }


def _record_rows(
    rows: List[CentroidRegionRow],
    *,
    rep,
    design_rep,
    delta_base_rep,
    budget: int,
    method: str,
    prompt_id: str,
    collection_mode: str,
    head_state: HeadState,
    train_qbank,
    holdout_qbank,
    changed_atom_fraction: float,
    assigned_token_fraction: float,
    mean_region_size: float,
    mean_assignment_similarity: float,
):
    train_queries, train_weights = train_qbank.get_weighted_bank()
    design = build_design(train_queries, train_weights, design_rep.support_keys, design_rep.betas)
    d_stats = matrix_stats(design)
    upd_stats = delta_stats(delta_base_rep, rep, design)
    for split_name, bank in (("train", train_qbank), ("holdout", holdout_qbank)):
        l_z, l_n, l_true, _l_lin = compute_metrics(rep, head_state, bank)
        rows.append(
            CentroidRegionRow(
                prompt_id=prompt_id,
                collection_mode=collection_mode,
                layer=head_state.layer_idx,
                kv_head=head_state.head_idx,
                budget=budget,
                method=method,
                split=split_name,
                l_true=l_true,
                l_z=l_z,
                l_n_per_dim=l_n,
                changed_atom_fraction=changed_atom_fraction,
                assigned_token_fraction=assigned_token_fraction,
                mean_region_size=mean_region_size,
                mean_assignment_similarity=mean_assignment_similarity,
                design_stable_rank=d_stats["stable_rank"],
                low_sv_delta_share=upd_stats["low_sv_delta_share"],
            )
        )


def summarize(rows: List[dict]) -> dict:
    holdout = [row for row in rows if row["split"] == "holdout"]
    grouped = defaultdict(list)
    for row in holdout:
        grouped[(row["collection_mode"], row["method"])].append(row)

    mean_rows = []
    for (mode, method), items in sorted(grouped.items()):
        mean_rows.append(
            {
                "collection_mode": mode,
                "method": method,
                "mean_holdout_l_true": sum(r["l_true"] for r in items) / len(items),
                "mean_changed_atom_fraction": sum(r["changed_atom_fraction"] for r in items) / len(items),
                "mean_assigned_token_fraction": sum(r["assigned_token_fraction"] for r in items) / len(items),
                "mean_region_size": sum(r["mean_region_size"] for r in items) / len(items),
                "mean_assignment_similarity": sum(r["mean_assignment_similarity"] for r in items) / len(items),
                "mean_design_stable_rank": sum(r["design_stable_rank"] for r in items) / len(items),
                "mean_low_sv_delta_share": sum(r["low_sv_delta_share"] for r in items) / len(items),
                "num_cells": len(items),
            }
        )

    by_cell = {}
    for row in holdout:
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
        ("anchor_region_centroid", "anchor_region_local"),
        ("anchor_region_centroid+vfit", "anchor_region_local+vfit"),
    ]:
        deltas = defaultdict(list)
        coh = defaultdict(list)
        srank = defaultdict(list)
        lowsv = defaultdict(list)
        for key, lhs_row in by_cell.items():
            if key[-1] != lhs:
                continue
            rhs_key = (*key[:-1], rhs)
            if rhs_key not in by_cell:
                continue
            rhs_row = by_cell[rhs_key]
            mode = key[1]
            deltas[mode].append(lhs_row["l_true"] - rhs_row["l_true"])
            coh[mode].append(lhs_row["mean_assignment_similarity"] - rhs_row["mean_assignment_similarity"])
            srank[mode].append(lhs_row["design_stable_rank"] - rhs_row["design_stable_rank"])
            lowsv[mode].append(lhs_row["low_sv_delta_share"] - rhs_row["low_sv_delta_share"])
        for mode, vals in sorted(deltas.items()):
            comparisons.append(
                {
                    "collection_mode": mode,
                    "lhs": lhs,
                    "rhs": rhs,
                    "mean_delta_holdout_l_true": sum(vals) / len(vals),
                    "improved_cells": sum(v < 0 for v in vals),
                    "num_cells": len(vals),
                    "mean_assignment_similarity_delta": sum(coh[mode]) / len(coh[mode]),
                    "mean_design_stable_rank_delta": sum(srank[mode]) / len(srank[mode]),
                    "mean_low_sv_delta_share_delta": sum(lowsv[mode]) / len(lowsv[mode]),
                }
            )

    return {
        "mean_holdout_l_true_by_method": mean_rows,
        "comparisons": comparisons,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"

    beta_cfg = BetaFitConfig(
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
        max_fit_queries=args.max_queries,
    )

    rows: List[CentroidRegionRow] = []
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
                skipped.append({"prompt_id": prompt_name, "collection_mode": collection_mode, "reason": str(exc)})
                print(f"  skipped: {exc}", flush=True)
                continue

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

                    queries, weights = train_qbank.get_weighted_bank()
                    weighted_design, _weighted_target = _build_mass_frame(
                        queries.float(),
                        head_state.keys.float(),
                        weights.float(),
                    )
                    centroids = _weighted_kmeans_centroids(
                        queries.float(),
                        weights.float(),
                        k=args.num_centroids,
                        iters=args.kmeans_iters,
                    )

                    for frac in args.budgets:
                        budget = max(1, int(seq_len * frac))
                        base_rep = hybrid_support_baseline(head_state, train_qbank, budget, config=LIVE_HYBRID_CFG)
                        anchor_indices = _match_support_to_source_indices(
                            head_state.keys.float(),
                            base_rep.support_keys.float(),
                        )

                        local_region = _build_anchor_region_atoms(
                            keys_f=head_state.keys.float(),
                            values_f=head_state.values.float(),
                            weighted_design=weighted_design,
                            anchor_indices=anchor_indices,
                            assignment_window=args.assignment_window,
                            assignment_distance_penalty=args.assignment_distance_penalty,
                            assignment_score_floor=args.assignment_score_floor,
                            max_neighbor_blend=args.max_neighbor_blend,
                        )
                        centroid_region = _centroid_conditioned_region_atoms(
                            keys_f=head_state.keys.float(),
                            values_f=head_state.values.float(),
                            anchor_indices=anchor_indices,
                            centroids=centroids,
                            assignment_window=args.assignment_window,
                            assignment_distance_penalty=args.assignment_distance_penalty,
                            assignment_score_floor=args.assignment_score_floor,
                            max_neighbor_blend=args.max_neighbor_blend,
                        )

                        local_rep = base_rep.__class__(
                            support_keys=local_region["support_keys"].to(dtype=head_state.keys.dtype, device=head_state.keys.device),
                            support_values=local_region["support_values"].to(dtype=head_state.values.dtype, device=head_state.values.device),
                            betas=base_rep.betas.clone(),
                        )
                        centroid_rep = base_rep.__class__(
                            support_keys=centroid_region["support_keys"].to(dtype=head_state.keys.dtype, device=head_state.keys.device),
                            support_values=centroid_region["support_values"].to(dtype=head_state.values.dtype, device=head_state.values.device),
                            betas=base_rep.betas.clone(),
                        )

                        local_v = refit_values(local_rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
                        centroid_v = refit_values(centroid_rep, head_state.keys, head_state.values, train_qbank, beta_cfg)

                        for rep, method, region in [
                            (local_rep, "anchor_region_local", local_region),
                            (local_v, "anchor_region_local+vfit", local_region),
                            (centroid_rep, "anchor_region_centroid", centroid_region),
                            (centroid_v, "anchor_region_centroid+vfit", centroid_region),
                        ]:
                            assigned = region["assignments"].ge(0)
                            mean_assignment_similarity = float(
                                region["assignment_scores"][assigned].mean().item()
                            ) if assigned.any() else 0.0
                            if method == "anchor_region_local":
                                design_rep = local_rep
                                delta_base_rep = local_rep
                            elif method == "anchor_region_local+vfit":
                                design_rep = local_rep
                                delta_base_rep = local_rep
                            elif method == "anchor_region_centroid":
                                design_rep = centroid_rep
                                delta_base_rep = centroid_rep
                            else:
                                design_rep = centroid_rep
                                delta_base_rep = centroid_rep
                            _record_rows(
                                rows,
                                rep=rep,
                                design_rep=design_rep,
                                delta_base_rep=delta_base_rep,
                                budget=budget,
                                method=method,
                                prompt_id=prompt_name,
                                collection_mode=collection_mode,
                                head_state=head_state,
                                train_qbank=train_qbank,
                                holdout_qbank=holdout_qbank,
                                changed_atom_fraction=_changed_atom_fraction(rep, head_state),
                                assigned_token_fraction=float(assigned.float().mean().item()),
                                mean_region_size=float(region["region_sizes"].float().mean().item()),
                                mean_assignment_similarity=mean_assignment_similarity,
                            )

    payload = {
        "args": vars(args),
        "skipped": skipped,
        "rows": [asdict(r) for r in rows],
        "summary": summarize([asdict(r) for r in rows]),
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved centroid-region precursor to {out}", flush=True)


if __name__ == "__main__":
    main()
