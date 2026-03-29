#!/usr/bin/env python3
"""
Targeted forensic sweep for unstable fixed-support value fitting.

Focuses on one (layer, kv_head, budget) cell and compares:
  - no refit
  - plain value LS
  - stronger ridge
  - interpolation back toward original values
  - larger query-bank regimes

This is intended to separate:
  - support geometry problems
  - supervision-limited local regression
  - value-fit instability along weakly constrained design directions
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from run_experiment import build_prompt, build_query_bank, extract_kv_and_queries, load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import attention_mass_baseline
from kv_operator_matching.config import BetaFitConfig, QueryBankConfig
from kv_operator_matching.objectives import compute_logits, compute_response, compute_z, loss_n
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import fit_values


@dataclass
class SweepRow:
    bank_max_queries: int
    variant: str
    layer: int
    kv_head: int
    budget: int
    train_queries: int
    holdout_queries: int
    train_l_true: float
    holdout_l_true: float
    train_l_z: float
    holdout_l_z: float
    train_l_n_per_dim: float
    holdout_l_n_per_dim: float
    fit_norm_over_base: float
    delta_over_base_mean: float
    delta_max: float
    design_rank: int
    design_cond: float
    stable_rank: float
    low_sv_delta_share: float
    top5_holdout_error_share: float
    median_holdout_query_error: float
    max_holdout_query_error: float


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
    design = alpha * weights.sqrt().unsqueeze(-1)
    return design


def fit_values_anchored(
    base_rep: CompactRepresentation,
    ref_keys: torch.Tensor,
    ref_values: torch.Tensor,
    query_bank: QueryBank,
    value_ridge: float,
    interpolation: float = 1.0,
) -> CompactRepresentation:
    cfg = BetaFitConfig(value_ridge=value_ridge)
    fitted_values = fit_values(
        base_rep.support_keys,
        base_rep.betas,
        ref_keys,
        ref_values,
        query_bank,
        cfg,
    )
    support_values = (
        base_rep.support_values * (1.0 - interpolation) + fitted_values * interpolation
    )
    return CompactRepresentation(
        support_keys=base_rep.support_keys,
        support_values=support_values,
        betas=base_rep.betas,
    )


def low_singular_delta_share(design: torch.Tensor, delta_values: torch.Tensor) -> float:
    _u, _s, vh = torch.linalg.svd(design, full_matrices=False)
    coeff = vh @ delta_values
    energy = coeff.pow(2).sum(dim=1)
    if energy.numel() == 0:
        return 0.0
    cutoff = max(1, energy.numel() // 4)
    low_energy = energy[-cutoff:].sum()
    return float((low_energy / energy.sum().clamp(min=1e-12)).item())


def matrix_stats(design: torch.Tensor):
    sv = torch.linalg.svdvals(design)
    smax = float(sv.max().item())
    smin = float(sv.min().item())
    return {
        "rank": int(torch.linalg.matrix_rank(design).item()),
        "cond": smax / max(smin, 1e-12),
        "stable_rank": float((design.square().sum() / max(smax * smax, 1e-12)).item()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--kv-head", type=int, default=0)
    p.add_argument("--budget-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--save-json", default="results/forensic_value_regime.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    prompt = build_prompt()
    kv_states, query_states = extract_kv_and_queries(model, tokenizer, prompt, args.device, [args.layer])

    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    keys = kv_states[args.layer]["keys"][args.kv_head]
    values = kv_states[args.layer]["values"][args.kv_head]
    head_state = HeadState(head_idx=args.kv_head, layer_idx=args.layer, keys=keys, values=values)
    budget = max(1, int(keys.shape[0] * args.budget_frac))

    bank_sizes = [256, 1024]
    variants = [
        ("base", 1e-4, 0.0),
        ("plain", 1e-4, 1.0),
        ("ridge_1e-2", 1e-2, 1.0),
        ("ridge_1e0", 1.0, 1.0),
        ("interp_025", 1e-4, 0.25),
        ("interp_050", 1e-4, 0.50),
        ("interp_075", 1e-4, 0.75),
        ("ridge1_interp050", 1.0, 0.50),
    ]

    rows: list[SweepRow] = []
    for bank_max in bank_sizes:
        torch.manual_seed(args.seed)
        bank_cfg = QueryBankConfig(max_queries=bank_max, weighting_scheme="uniform")
        qbank = build_query_bank(query_states[args.layer], args.kv_head, n_heads, n_kv_heads, bank_cfg)
        train_bank, holdout_bank = qbank.split_train_holdout(args.train_fraction)
        base_rep = attention_mass_baseline(head_state, train_bank, budget)

        train_queries, train_weights = train_bank.get_weighted_bank()
        design = build_design(train_queries, train_weights, base_rep.support_keys, base_rep.betas)
        d_stats = matrix_stats(design)

        for variant, ridge, interpolation in variants:
            if variant == "base":
                rep = base_rep
            else:
                rep = fit_values_anchored(
                    base_rep,
                    keys,
                    values,
                    train_bank,
                    value_ridge=ridge,
                    interpolation=interpolation,
                )

            train_l_true, train_l_z, train_l_n = compute_metrics(rep, head_state, train_bank)
            holdout_l_true, holdout_l_z, holdout_l_n = compute_metrics(rep, head_state, holdout_bank)
            err_stats = per_query_error_stats(rep, head_state, holdout_bank)

            if variant == "base":
                fit_over_base = 1.0
                delta_over_base_mean = 0.0
                delta_max = 0.0
                low_sv_share = 0.0
            else:
                base_norm = torch.linalg.vector_norm(base_rep.support_values, dim=1)
                new_norm = torch.linalg.vector_norm(rep.support_values, dim=1)
                delta = torch.linalg.vector_norm(rep.support_values - base_rep.support_values, dim=1)
                fit_over_base = float(new_norm.mean().item() / base_norm.mean().clamp(min=1e-12).item())
                delta_over_base_mean = float((delta / base_norm.clamp(min=1e-12)).mean().item())
                delta_max = float(delta.max().item())
                low_sv_share = low_singular_delta_share(design, rep.support_values - base_rep.support_values)

            rows.append(SweepRow(
                bank_max_queries=bank_max,
                variant=variant,
                layer=args.layer,
                kv_head=args.kv_head,
                budget=budget,
                train_queries=len(train_bank),
                holdout_queries=len(holdout_bank),
                train_l_true=train_l_true,
                holdout_l_true=holdout_l_true,
                train_l_z=train_l_z,
                holdout_l_z=holdout_l_z,
                train_l_n_per_dim=train_l_n,
                holdout_l_n_per_dim=holdout_l_n,
                fit_norm_over_base=fit_over_base,
                delta_over_base_mean=delta_over_base_mean,
                delta_max=delta_max,
                design_rank=d_stats["rank"],
                design_cond=d_stats["cond"],
                stable_rank=d_stats["stable_rank"],
                low_sv_delta_share=low_sv_share,
                top5_holdout_error_share=err_stats["top5_share"],
                median_holdout_query_error=err_stats["median"],
                max_holdout_query_error=err_stats["max"],
            ))

    rows = sorted(rows, key=lambda r: (r.bank_max_queries, r.holdout_l_true))
    for row in rows:
        print(
            f"bank={row.bank_max_queries:4d} {row.variant:<14} "
            f"holdout L_true={row.holdout_l_true:8.3f} "
            f"holdout L_N/d={row.holdout_l_n_per_dim:8.3f} "
            f"fit/base={row.fit_norm_over_base:6.2f} "
            f"delta/base={row.delta_over_base_mean:6.2f} "
            f"low-sv-share={row.low_sv_delta_share:6.2f}"
        )

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
