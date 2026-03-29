#!/usr/bin/env python3
"""
Qwen 2.5 N/Z operator matching smoke test.

Supports two query-bank collection modes:
  prefill - run a single forward pass on a prompt and use the prompt queries
            as an offline proxy bank
  online  - prefill the prompt into the KV cache, then collect decode-step
            queries during greedy generation into a live query bank

Methods compared:
  recency         - keep last m tokens, betas=1
  recency+refit   - recency support, NNLS beta-refit
  recency+vfit    - recency support, anchored value-only refit
  recency+phase1b - recency support, safe sequential mass beta-fit then value refit
  attn_mass       - highest attention-mass tokens, betas=1
  attn_mass+refit - attn_mass support, NNLS beta-refit
  attn_mass+vfit  - attn_mass support, anchored value-only refit
  attn_mass+phase1b - attn_mass support, safe sequential mass beta-fit then value refit
  uniform         - random selection, betas=1 (sanity check)

Metrics reported per (layer, kv_head, budget):
  L_Z     - partition function error  (normalized by total query weight)
  L_N/dim - numerator error per dim   (normalized by total weight * d_v)
  L_true  - true response error       (normalized by total query weight)

Usage:
  python run_experiment.py [--model MODEL] [--layers L L L] [--budgets B B]

Adapted from kv_compaction_experiment patterns; see docs/old_repo_relationship.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from importlib.machinery import ModuleSpec
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kv_operator_matching.baselines import (
    attention_mass_baseline,
    recency_baseline,
    uniform_baseline,
)
from kv_operator_matching.beta_fit import refit_beta, sequential_refit_beta_and_values
from kv_operator_matching.config import BetaFitConfig, QueryBankConfig
from kv_operator_matching.objectives import compute_logits, compute_response, loss_lin, loss_n, loss_true_response, loss_z
from kv_operator_matching.query_bank import QueryBank
from kv_operator_matching.types import CompactRepresentation, HeadState
from kv_operator_matching.value_fit import refit_values


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    method: str
    layer: int
    kv_head: int
    budget: int
    n_tokens: int
    split: str
    n_queries: int
    l_z: float    # log-scale mass error: (log Z_hat - log Z_ref)^2 normalized by W
    l_n_per_dim: float
    l_true: float # response error: ||A_hat - A_ref||^2 normalized by W
    l_lin: float
    elapsed_ms: float


@dataclass
class RefitDiagnostic:
    method: str
    layer: int
    kv_head: int
    budget: int
    train_queries: int
    holdout_queries: int
    pre_train_l_lin: float
    post_train_l_lin: float
    pre_holdout_l_lin: float
    post_holdout_l_lin: float
    pre_holdout_l_true: float
    post_holdout_l_true: float
    beta_min: float
    beta_max: float
    beta_mean: float


@dataclass
class GoodSupportDiagnostic:
    layer: int
    kv_head: int
    budget: int
    method: str
    holdout_l_true: float
    design_rank: int
    design_stable_rank: float
    design_condition_number: float
    holdout_top5_error_share: float
    passed: bool


# ---------------------------------------------------------------------------
# Model loading and KV/query extraction
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    # Some local environments have a broken boto3/botocore stack that causes
    # transformers -> accelerate import-time failures, even though this script
    # does not use SageMaker integration. Install a stub only when real boto3
    # is unavailable so the smoke test can still run.
    try:
        import boto3  # type: ignore  # noqa: F401
    except Exception:
        boto3_stub = types.ModuleType("boto3")
        boto3_stub.__spec__ = ModuleSpec("boto3", loader=None)
        sys.modules.setdefault("boto3", boto3_stub)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_params:.1f}B params on {device}", flush=True)
    return model, tokenizer


def _extract_kv_states_from_cache(cache, layers_of_interest: List[int]) -> Dict:
    """Return CPU float32 KV states for the requested layers from a model cache."""
    kv_states: Dict = {}
    for layer_idx in layers_of_interest:
        if hasattr(cache, "key_cache"):   # DynamicCache (transformers >= 4.38)
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
        else:                              # legacy tuple-of-tuples
            k, v = cache[layer_idx]
        kv_states[layer_idx] = {
            "keys": k[0].float().cpu(),
            "values": v[0].float().cpu(),
        }
    return kv_states


def _compute_query_states_from_hidden_states(
    model,
    hidden_states_list,
    position_ids: torch.Tensor,
    layers_of_interest: List[int],
) -> Dict:
    """Compute post-RoPE query states from per-layer input hidden states."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError:
        from transformers.models.qwen2_5.modeling_qwen2_5 import apply_rotary_pos_emb

    query_states: Dict = {}
    for layer_idx in layers_of_interest:
        h = hidden_states_list[layer_idx].float()
        attn = model.model.layers[layer_idx].self_attn
        bsz, slen, _ = h.shape
        n_heads = getattr(attn, "num_heads", None) or getattr(model.config, "num_attention_heads")
        n_kv_heads = (
            getattr(attn, "num_key_value_heads", None)
            or getattr(model.config, "num_key_value_heads")
        )
        head_dim = attn.head_dim

        model_dtype = next(attn.q_proj.parameters()).dtype
        h_proj = h.to(model_dtype)

        with torch.no_grad():
            q_raw = attn.q_proj(h_proj)
            k_raw = attn.k_proj(h_proj)

        q_raw = q_raw.view(bsz, slen, n_heads, head_dim).transpose(1, 2)
        k_raw = k_raw.view(bsz, slen, n_kv_heads, head_dim).transpose(1, 2)

        import inspect
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = model.model.rotary_emb
        rotary_sig = inspect.signature(rotary_emb.forward)
        rotary_params = list(rotary_sig.parameters.keys())
        if "seq_len" in rotary_params:
            cos, sin = rotary_emb(q_raw, seq_len=slen)
            q_rope, _ = apply_rotary_pos_emb(q_raw, k_raw, cos, sin, position_ids)
        else:
            cos, sin = rotary_emb(k_raw, position_ids)
            q_rope, _ = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)

        query_states[layer_idx] = q_rope[0].float().cpu()

    return query_states


def extract_kv_and_queries(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layers_of_interest: List[int],
) -> Tuple[Dict, Dict]:
    """
    Run a single prefill pass and return per-layer KV states and query vectors.

    KV states: from past_key_values (post-RoPE, as used at inference time).
    Query states: computed by manually applying q_proj + RoPE to each layer's
    input hidden state (from output_hidden_states=True), giving the actual
    post-RoPE query vectors that would attend to the cached keys.

    Returns:
        kv_states:    {layer: {"keys": (n_kv_heads, seq, head_dim),
                               "values": (n_kv_heads, seq, head_dim)}}
        query_states: {layer: (n_heads, seq, head_dim)}
    Both are float32 on CPU.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {seq_len} tokens", flush=True)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)

    kv_states = _extract_kv_states_from_cache(outputs.past_key_values, layers_of_interest)
    query_states = _compute_query_states_from_hidden_states(
        model,
        outputs.hidden_states,
        torch.arange(seq_len, device=device).unsqueeze(0),
        layers_of_interest,
    )
    return kv_states, query_states


# ---------------------------------------------------------------------------
# Query bank construction (context-prefill strategy)
# ---------------------------------------------------------------------------

def _forward_with_cache_position(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    start_position: int,
    *,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
):
    """Run a cache-aware forward pass with explicit logical positions."""
    seq_len = input_ids.shape[1]
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
        "output_attentions": output_attentions,
    }
    if seq_len > 0:
        kwargs["cache_position"] = torch.arange(
            start_position,
            start_position + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )
    with torch.no_grad():
        try:
            return model(**kwargs)
        except TypeError:
            kwargs.pop("cache_position", None)
            return model(**kwargs)


def _select_greedy_non_eos(logits: torch.Tensor, eos_token_id: Optional[int]) -> int:
    """Greedy decode, but avoid an immediate EOS when another high-logit token exists."""
    ranked = torch.argsort(logits[0], descending=True)
    for token_id in ranked.tolist():
        if eos_token_id is None or token_id != eos_token_id:
            return int(token_id)
    return int(ranked[0].item())


def _build_empty_query_banks(
    layers_of_interest: List[int],
    n_kv_heads: int,
    config: QueryBankConfig,
) -> Dict[Tuple[int, int], QueryBank]:
    return {
        (layer_idx, kv_head_idx): QueryBank(config)
        for layer_idx in layers_of_interest
        for kv_head_idx in range(n_kv_heads)
    }


def _add_query_states_to_banks(
    query_states: Dict[int, torch.Tensor],
    query_banks: Dict[Tuple[int, int], QueryBank],
    n_heads: int,
    n_kv_heads: int,
) -> None:
    """Append query states to the per-(layer, kv_head) rolling banks."""
    qpk = n_heads // n_kv_heads
    for layer_idx, layer_queries in query_states.items():
        for kv_head_idx in range(n_kv_heads):
            first = kv_head_idx * qpk
            head_queries = layer_queries[first : first + qpk].reshape(-1, layer_queries.shape[-1])
            query_banks[(layer_idx, kv_head_idx)].add_queries(head_queries)


def collect_online_kv_and_query_banks(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layers_of_interest: List[int],
    bank_config: QueryBankConfig,
    max_new_tokens: int,
    prefill_chunk_size: int,
) -> Tuple[Dict, Dict[Tuple[int, int], QueryBank], int]:
    """
    Prefill the prompt into cache, then collect decode-step query evidence.

    Returns:
        kv_states: final per-layer KV cache state after generation
        query_banks: rolling query bank for each (layer, kv_head)
        generated_tokens: number of decode steps captured
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt: {prompt_len} tokens", flush=True)

    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    query_banks = _build_empty_query_banks(layers_of_interest, n_kv_heads, bank_config)

    past_key_values = None
    logits = None
    processed = 0
    chunk_size = max(1, prefill_chunk_size)
    while processed < prompt_len:
        chunk_end = min(prompt_len, processed + chunk_size)
        chunk_ids = input_ids[:, processed:chunk_end]
        outputs = _forward_with_cache_position(
            model,
            chunk_ids,
            past_key_values,
            processed,
            output_hidden_states=False,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        processed = chunk_end

    if logits is None:
        raise RuntimeError("Prompt prefill did not produce logits.")

    generated_tokens = 0
    logical_position = prompt_len
    eos_token_id = tokenizer.eos_token_id
    next_token = _select_greedy_non_eos(logits, eos_token_id)
    while generated_tokens < max_new_tokens:
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
        outputs = _forward_with_cache_position(
            model,
            token_tensor,
            past_key_values,
            logical_position,
            output_hidden_states=True,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        query_states = _compute_query_states_from_hidden_states(
            model,
            outputs.hidden_states,
            torch.tensor([[logical_position]], device=device, dtype=torch.long),
            layers_of_interest,
        )
        _add_query_states_to_banks(query_states, query_banks, n_heads, n_kv_heads)
        generated_tokens += 1
        logical_position += 1
        next_token = _select_greedy_non_eos(logits, eos_token_id)

    if generated_tokens == 0:
        raise RuntimeError("Online collection generated zero observed decode steps.")

    kv_states = _extract_kv_states_from_cache(past_key_values, layers_of_interest)
    return kv_states, query_banks, generated_tokens

def build_query_bank(
    query_states_for_layer: torch.Tensor,  # (n_heads, seq_len, head_dim)
    kv_head_idx: int,
    n_heads: int,
    n_kv_heads: int,
    config: QueryBankConfig,
) -> QueryBank:
    """
    Build a query bank for one KV head from context-prefill query vectors.

    In GQA, kv_head_idx is shared by (n_heads // n_kv_heads) query heads.
    We stack all their prefill query vectors into the bank with uniform weights.
    """
    qpk = n_heads // n_kv_heads          # query heads per KV head
    first = kv_head_idx * qpk
    head_queries = query_states_for_layer[first : first + qpk]  # (qpk, seq, head_dim)
    all_queries = head_queries.reshape(-1, head_queries.shape[-1])   # (qpk*seq, head_dim)

    # Subsample uniformly so the bank represents all query heads and positions
    # equally. The QueryBank uses FIFO eviction, which would keep only the last
    # max_queries entries (biased to the last query head) if we add all at once.
    n_total = all_queries.shape[0]
    max_q = config.max_queries
    if max_q > 0 and n_total > max_q:
        idx = torch.randperm(n_total)[:max_q]
        all_queries = all_queries[idx]

    bank = QueryBank(config)
    bank.add_queries(all_queries)
    return bank


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    rep: CompactRepresentation,
    head_state: HeadState,
    qbank: QueryBank,
) -> Tuple[float, float, float, float]:
    """Return (l_z, l_n_per_dim, l_true, l_lin) normalized by total query weight."""
    queries, weights = qbank.get_weighted_bank()
    W  = weights.sum().clamp(min=1e-8).item()
    dv = head_state.values.shape[-1]

    with torch.no_grad():
        lz = loss_z(
            queries, weights,
            rep.support_keys, rep.betas,
            head_state.keys,
        ).item() / W

        ln = loss_n(
            queries, weights,
            rep.support_keys, rep.support_values, rep.betas,
            head_state.keys, head_state.values,
        ).item() / (W * dv)

        lt = loss_true_response(
            queries, weights,
            rep.support_keys, rep.support_values, rep.betas,
            head_state.keys, head_state.values,
        ).item() / W

        llin = loss_lin(
            queries, weights,
            rep.support_keys, rep.support_values, rep.betas,
            head_state.keys, head_state.values,
        ).item() / W

    return lz, ln, lt, llin


def compute_error_concentration(
    rep: CompactRepresentation,
    head_state: HeadState,
    qbank: QueryBank,
    topk: int = 5,
) -> float:
    """Return the share of weighted response error carried by the top-k queries."""
    queries, weights = qbank.get_weighted_bank()
    with torch.no_grad():
        a_hat = compute_response(queries, rep.support_keys, rep.support_values, rep.betas)
        a_ref = compute_response(queries, head_state.keys, head_state.values)
        weighted = ((a_hat - a_ref) ** 2).sum(dim=-1) * weights
        total = weighted.sum().clamp(min=1e-12)
        return (weighted.topk(min(topk, weighted.numel())).values.sum() / total).item()


def compute_design_stats(
    rep: CompactRepresentation,
    qbank: QueryBank,
) -> tuple[int, float, float]:
    """Return (rank, stable_rank, condition_number) for the compact attention design."""
    queries, weights = qbank.get_weighted_bank()
    with torch.no_grad():
        logits = compute_logits(queries, rep.support_keys) + torch.log(
            rep.betas.clamp(min=1e-30)
        ).unsqueeze(0)
        alpha = torch.softmax(logits, dim=-1)
        design = alpha * weights.sqrt().unsqueeze(-1)
        sv = torch.linalg.svdvals(design)
        smax = sv.max().item()
        smin = sv.min().item()
        rank = int(torch.linalg.matrix_rank(design).item())
        stable_rank = float((design.square().sum() / max(smax * smax, 1e-12)).item())
        cond = float(smax / max(smin, 1e-12))
    return rank, stable_rank, cond


def evaluate_good_support(
    rep: CompactRepresentation,
    head_state: HeadState,
    train_qbank: QueryBank,
    holdout_qbank: QueryBank,
    layer: int,
    kv_head: int,
    budget: int,
    args: argparse.Namespace,
) -> GoodSupportDiagnostic:
    """Evaluate whether a support passes the predeclared good-support gate."""
    _lz, _ln, holdout_l_true, _llin = compute_metrics(rep, head_state, holdout_qbank)
    rank, stable_rank, cond = compute_design_stats(rep, train_qbank)
    top5_share = compute_error_concentration(rep, head_state, holdout_qbank)
    passed = (
        holdout_l_true <= args.good_support_ltrue_threshold
        and stable_rank >= args.good_support_stable_rank_threshold
        and cond <= args.good_support_cond_threshold
        and top5_share <= args.good_support_top5_share_threshold
    )
    return GoodSupportDiagnostic(
        layer=layer,
        kv_head=kv_head,
        budget=budget,
        method="attn_mass",
        holdout_l_true=holdout_l_true,
        design_rank=rank,
        design_stable_rank=stable_rank,
        design_condition_number=cond,
        holdout_top5_error_share=top5_share,
        passed=passed,
    )


def run_methods(
    head_state: HeadState,
    train_qbank: QueryBank,
    holdout_qbank: QueryBank,
    budget: int,
    beta_cfg: BetaFitConfig,
    seed: int,
    mode: str,
) -> Tuple[List[MethodResult], List[RefitDiagnostic]]:
    """Run all methods for one (layer, kv_head, budget) combination."""
    results = []
    diagnostics = []
    torch.manual_seed(seed)
    l, h = head_state.layer_idx, head_state.head_idx
    n = head_state.keys.shape[0]

    def rec(method: str, rep: CompactRepresentation, t0: float):
        elapsed_ms = (time.time() - t0) * 1000
        for split_name, bank in (("train", train_qbank), ("holdout", holdout_qbank)):
            lz, ln, lt, llin = compute_metrics(rep, head_state, bank)
            results.append(MethodResult(
                method=method,
                layer=l,
                kv_head=h,
                budget=budget,
                n_tokens=n,
                split=split_name,
                n_queries=len(bank),
                l_z=lz,
                l_n_per_dim=ln,
                l_true=lt,
                l_lin=llin,
                elapsed_ms=elapsed_ms,
            ))

    def add_refit_diag(
        method: str,
        pre_rep: CompactRepresentation,
        post_rep: CompactRepresentation,
    ):
        _pre_train_lz, _pre_train_ln, pre_train_lt, pre_train_llin = compute_metrics(pre_rep, head_state, train_qbank)
        _post_train_lz, _post_train_ln, post_train_lt, post_train_llin = compute_metrics(post_rep, head_state, train_qbank)
        _pre_holdout_lz, _pre_holdout_ln, pre_holdout_lt, pre_holdout_llin = compute_metrics(pre_rep, head_state, holdout_qbank)
        _post_holdout_lz, _post_holdout_ln, post_holdout_lt, post_holdout_llin = compute_metrics(post_rep, head_state, holdout_qbank)
        diagnostics.append(RefitDiagnostic(
            method=method,
            layer=l,
            kv_head=h,
            budget=budget,
            train_queries=len(train_qbank),
            holdout_queries=len(holdout_qbank),
            pre_train_l_lin=pre_train_llin,
            post_train_l_lin=post_train_llin,
            pre_holdout_l_lin=pre_holdout_llin,
            post_holdout_l_lin=post_holdout_llin,
            pre_holdout_l_true=pre_holdout_lt,
            post_holdout_l_true=post_holdout_lt,
            beta_min=float(post_rep.betas.min().item()),
            beta_max=float(post_rep.betas.max().item()),
            beta_mean=float(post_rep.betas.mean().item()),
        ))

    t0 = time.time()
    if mode == "full":
        rep = recency_baseline(head_state, budget)
        rec("recency", rep, t0)

        t0 = time.time()
        rep_r = refit_beta(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
        rec("recency+refit", rep_r, t0)
        add_refit_diag("recency+refit", rep, rep_r)

        t0 = time.time()
        rep_v = refit_values(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
        rec("recency+vfit", rep_v, t0)
        add_refit_diag("recency+vfit", rep, rep_v)

        t0 = time.time()
        rep_p = sequential_refit_beta_and_values(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
        rec("recency+phase1b", rep_p, t0)
        add_refit_diag("recency+phase1b", rep, rep_p)

    t0 = time.time()
    rep = attention_mass_baseline(head_state, train_qbank, budget)
    rec("attn_mass", rep, t0)

    t0 = time.time()
    rep_r = refit_beta(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
    rec("attn_mass+refit", rep_r, t0)
    add_refit_diag("attn_mass+refit", rep, rep_r)

    t0 = time.time()
    rep_v = refit_values(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
    rec("attn_mass+vfit", rep_v, t0)
    add_refit_diag("attn_mass+vfit", rep, rep_v)

    t0 = time.time()
    rep_p = sequential_refit_beta_and_values(rep, head_state.keys, head_state.values, train_qbank, beta_cfg)
    rec("attn_mass+phase1b", rep_p, t0)
    add_refit_diag("attn_mass+phase1b", rep, rep_p)

    if mode == "full":
        t0 = time.time()
        rep = uniform_baseline(head_state, budget)
        rec("uniform", rep, t0)

    return results, diagnostics


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

FULL_METHODS = [
    "recency",
    "recency+refit",
    "recency+vfit",
    "recency+phase1b",
    "attn_mass",
    "attn_mass+refit",
    "attn_mass+vfit",
    "attn_mass+phase1b",
    "uniform",
]
FOCUSED_METHODS = [
    "attn_mass",
    "attn_mass+refit",
    "attn_mass+vfit",
    "attn_mass+phase1b",
]
MW = max(len(m) for m in FULL_METHODS) + 2

def print_results(results: List[MethodResult], method_order: List[str]):
    # Group by (layer, kv_head, budget, n_tokens, split)
    groups: Dict = {}
    for r in results:
        key = (r.layer, r.kv_head, r.budget, r.n_tokens, r.split)
        groups.setdefault(key, {})[r.method] = r

    for (layer, kv_head, budget, n_tokens, split), mmap in sorted(groups.items()):
        frac = budget / n_tokens
        print(f"\nLayer {layer:2d}  KV-head {kv_head}  "
              f"budget {budget}/{n_tokens} ({frac:.0%})  split={split}")
        print(f"  {'Method':<{MW}}  {'L_Z':>10}  {'L_true':>10}  {'L_lin':>10}  {'ms':>7}")
        print(f"  {'-'*MW}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}")
        for method in method_order:
            r = mmap.get(method)
            if r is None:
                continue
            print(
                f"  {method:<{MW}}  {r.l_z:>10.4f}  {r.l_true:>10.4f}  "
                f"{r.l_lin:>10.4f}  {r.elapsed_ms:>7.1f}"
            )


def print_refit_diagnostics(diagnostics: List[RefitDiagnostic]):
    if not diagnostics:
        return
    print("\nRefit diagnostics (holdout is the signal to trust):")
    for diag in sorted(diagnostics, key=lambda d: (d.layer, d.kv_head, d.budget, d.method)):
        print(
            f"  L={diag.layer:2d} H={diag.kv_head} {diag.method:<15} "
            f"train L_lin {diag.pre_train_l_lin:.4f}->{diag.post_train_l_lin:.4f}  "
            f"holdout L_lin {diag.pre_holdout_l_lin:.4f}->{diag.post_holdout_l_lin:.4f}  "
            f"holdout L_true {diag.pre_holdout_l_true:.4f}->{diag.post_holdout_l_true:.4f}  "
            f"beta[{diag.beta_min:.3f}, {diag.beta_max:.3f}] mean={diag.beta_mean:.3f}"
        )


def print_good_support_diagnostics(diagnostics: List[GoodSupportDiagnostic]):
    if not diagnostics:
        return
    print("\nGood-support gate:")
    for diag in sorted(diagnostics, key=lambda d: (d.layer, d.kv_head, d.budget)):
        status = "PASS" if diag.passed else "FAIL"
        print(
            f"  {status} L={diag.layer:2d} H={diag.kv_head} budget={diag.budget:<3d} "
            f"holdout L_true={diag.holdout_l_true:.4f} "
            f"rank={diag.design_rank:<3d} "
            f"stable_rank={diag.design_stable_rank:.2f} "
            f"cond={diag.design_condition_number:.2e} "
            f"top5_share={diag.holdout_top5_error_share:.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",   default="Qwen/Qwen2.5-3B")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--collection-mode", choices=["prefill", "online"], default="online")
    p.add_argument("--query-weighting", choices=["uniform", "recency"], default=None)
    p.add_argument("--layers",  nargs="+", type=int,   default=[4, 12, 20, 28])
    p.add_argument("--budgets", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument("--mode", choices=["full", "focused"], default="focused")
    p.add_argument("--max-queries", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--good-support-ltrue-threshold", type=float, default=10.0)
    p.add_argument("--good-support-stable-rank-threshold", type=float, default=8.0)
    p.add_argument("--good-support-cond-threshold", type=float, default=1e8)
    p.add_argument("--good-support-top5-share-threshold", type=float, default=0.5)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def build_prompt() -> str:
    """Concatenate a few smoke-test turns for a medium-length prompt."""
    data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    texts = []
    for fname in sorted(data_path.glob("*.json"))[:2]:
        try:
            data = json.loads(fname.read_text())
            for sample in data.get("samples", [])[:1]:
                turns = sample.get("turns", [])
                texts.append("\n\n".join(
                    f"[{t['speaker'].upper()}]: {t['content']}"
                    for t in turns[:8] if t.get("content")
                ))
        except Exception:
            pass
    return ("\n\n---\n\n".join(texts) if texts else
            "Describe the key design decisions, failure modes, and recovery procedures "
            "for a distributed warehouse management system with partial fault tolerance.")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model, args.device)
    n_heads    = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    print(f"  Heads: {n_heads} query / {n_kv_heads} KV  "
          f"(GQA {n_heads // n_kv_heads}x per KV head)")

    prompt = build_prompt()
    weighting_scheme = args.query_weighting or (
        "recency" if args.collection_mode == "online" else "uniform"
    )
    bank_cfg = QueryBankConfig(max_queries=args.max_queries, weighting_scheme=weighting_scheme)
    print(f"\nCollecting KV + queries for layers {args.layers} ...", flush=True)
    t0 = time.time()
    query_banks: Dict[Tuple[int, int], QueryBank]
    if args.collection_mode == "prefill":
        kv_states, query_states = extract_kv_and_queries(
            model, tokenizer, prompt, args.device, args.layers
        )
        query_banks = {}
        for layer_idx in args.layers:
            for kv_head in range(n_kv_heads):
                query_banks[(layer_idx, kv_head)] = build_query_bank(
                    query_states[layer_idx], kv_head, n_heads, n_kv_heads, bank_cfg
                )
        print("  Collection mode: prefill proxy", flush=True)
    else:
        kv_states, query_banks, generated_tokens = collect_online_kv_and_query_banks(
            model,
            tokenizer,
            prompt,
            args.device,
            args.layers,
            bank_cfg,
            max_new_tokens=args.max_new_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        print(f"  Collection mode: online decode ({generated_tokens} observed tokens)", flush=True)
    print(f"  Query weighting: {weighting_scheme}", flush=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    beta_cfg = BetaFitConfig(
        normalize_lin=True,
        ridge=1e-4,
        value_ridge=1.0,
        value_interpolation=0.5,
    )
    all_results: List[MethodResult] = []
    all_diagnostics: List[RefitDiagnostic] = []
    all_support_diagnostics: List[GoodSupportDiagnostic] = []

    for layer_idx in args.layers:
        seq_len = kv_states[layer_idx]["keys"].shape[1]

        for kv_head in range(n_kv_heads):
            qbank = query_banks[(layer_idx, kv_head)]
            train_qbank, holdout_qbank = qbank.split_train_holdout(args.train_fraction)
            keys   = kv_states[layer_idx]["keys"][kv_head]    # (seq_len, head_dim)
            values = kv_states[layer_idx]["values"][kv_head]
            head_state = HeadState(
                head_idx=kv_head, layer_idx=layer_idx, keys=keys, values=values
            )

            for frac in args.budgets:
                budget = max(1, int(seq_len * frac))
                beta_cfg.support_size = budget
                n_qbank = len(qbank)
                attn_rep = attention_mass_baseline(head_state, train_qbank, budget)
                support_diag = evaluate_good_support(
                    attn_rep,
                    head_state,
                    train_qbank,
                    holdout_qbank,
                    layer_idx,
                    kv_head,
                    budget,
                    args,
                )
                all_support_diagnostics.append(support_diag)
                if args.mode == "focused" and not support_diag.passed:
                    print(
                        f"  L={layer_idx:2d} H={kv_head} budget={budget}/{seq_len}({frac:.0%}) "
                        f"qbank={n_qbank} train={len(train_qbank)} holdout={len(holdout_qbank)} "
                        f"good_support=FAIL -> skipped",
                        flush=True,
                    )
                    continue

                print(
                    f"  L={layer_idx:2d} H={kv_head} budget={budget}/{seq_len}({frac:.0%}) "
                    f"qbank={n_qbank} train={len(train_qbank)} holdout={len(holdout_qbank)} "
                    f"good_support={'PASS' if support_diag.passed else 'FAIL'}",
                    flush=True,
                )

                res, diag = run_methods(
                    head_state,
                    train_qbank,
                    holdout_qbank,
                    budget,
                    beta_cfg,
                    args.seed,
                    args.mode,
                )
                all_results.extend(res)
                all_diagnostics.extend(diag)

    print("\n" + "=" * 70)
    print_results(all_results, FOCUSED_METHODS if args.mode == "focused" else FULL_METHODS)
    print_good_support_diagnostics(all_support_diagnostics)
    print_refit_diagnostics(all_diagnostics)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(
            json.dumps(
                {
                    "results": [asdict(r) for r in all_results],
                    "good_support_diagnostics": [asdict(d) for d in all_support_diagnostics],
                    "refit_diagnostics": [asdict(d) for d in all_diagnostics],
                },
                indent=2,
            )
        )
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
