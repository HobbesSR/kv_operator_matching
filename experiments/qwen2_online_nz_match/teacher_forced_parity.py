#!/usr/bin/env python3
"""
Parity checks for suffix teacher-forced decode collection.

This script compares two ways of obtaining continuation-side query states:

1. teacher-forced-suffix decode: feed continuation tokens one by one with a live cache
2. batched continuation prefill: feed the same continuation tokens as a chunk

For a causal transformer, these should agree up to numerical noise when cache
positions and KV handling are correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_experiment import (
    _compute_query_states_from_hidden_states,
    _extract_kv_states_from_cache,
    _forward_with_captured_layer_inputs,
    _forward_with_cache_position,
    extract_teacher_forced_continuation_ids,
    load_model,
    load_prompt_segments_from_file,
)


@dataclass
class TensorParity:
    layer: int
    name: str
    max_abs: float
    mean_abs: float
    allclose_atol_1e_4: bool


@dataclass
class TeacherForcedParityResult:
    prompt_file: str
    prefix_turns: int
    continuation_turns: int
    observed_tokens: int
    teacher_forced_continuation_tokens: int
    boundary_kv_parity: List[TensorParity]
    query_parity: List[TensorParity]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt-file", default="near_capacity_dispatch_safe.json")
    p.add_argument("--prefix-turns", type=int, default=8)
    p.add_argument("--continuation-turns", type=int, default=4)
    p.add_argument("--layers", nargs="+", type=int, default=[4, 20])
    p.add_argument("--max-continuation-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=64)
    p.add_argument("--save-json", default="results/scratch/teacher_forced_parity.json")
    return p.parse_args()


def _parity_stats(layer: int, name: str, a: torch.Tensor, b: torch.Tensor) -> TensorParity:
    diff = (a.float() - b.float()).abs()
    return TensorParity(
        layer=layer,
        name=name,
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        allclose_atol_1e_4=bool(torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)),
    )


def _prefill_prompt_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layers: List[int],
    chunk_size: int,
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    past_key_values = None
    processed = 0
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
        processed = chunk_end
    boundary_kv = _extract_kv_states_from_cache(past_key_values, layers)
    return input_ids, past_key_values, boundary_kv


def _teacher_forced_query_states(
    model,
    past_key_values,
    continuation_token_ids: List[int],
    device: str,
    layers: List[int],
    prompt_len: int,
):
    per_layer_steps: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    logical_position = prompt_len
    for token_id in continuation_token_ids:
        token_tensor = torch.tensor([[int(token_id)]], device=device, dtype=torch.long)
        outputs, layer_inputs = _forward_with_captured_layer_inputs(
            model,
            token_tensor,
            past_key_values,
            logical_position,
            layers,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values
        qstates = _compute_query_states_from_hidden_states(
            model,
            layer_inputs,
            torch.tensor([[logical_position]], device=device, dtype=torch.long),
            layers,
        )
        for layer in layers:
            per_layer_steps[layer].append(qstates[layer][:, 0:1, :].cpu())
        logical_position += 1

    stacked = {
        layer: torch.cat(per_layer_steps[layer], dim=1) if per_layer_steps[layer] else torch.empty(0)
        for layer in layers
    }
    final_cache = past_key_values
    return stacked, final_cache


def _batched_continuation_query_states(
    model,
    past_key_values,
    continuation_token_ids: List[int],
    device: str,
    layers: List[int],
    prompt_len: int,
):
    cont_tensor = torch.tensor([continuation_token_ids], device=device, dtype=torch.long)
    outputs, layer_inputs = _forward_with_captured_layer_inputs(
        model,
        cont_tensor,
        past_key_values,
        prompt_len,
        layers,
        output_attentions=False,
    )
    position_ids = torch.arange(
        prompt_len,
        prompt_len + len(continuation_token_ids),
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    qstates = _compute_query_states_from_hidden_states(
        model,
        layer_inputs,
        position_ids,
        layers,
    )
    return qstates, outputs.past_key_values


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.device)

    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = smoke_data_path / args.prompt_file
    prompt, continuation_text = load_prompt_segments_from_file(
        prompt_path,
        prefix_turns=args.prefix_turns,
        continuation_turns=args.continuation_turns,
    )
    continuation_token_ids = extract_teacher_forced_continuation_ids(
        tokenizer,
        prompt,
        continuation_text,
    )[: args.max_continuation_tokens]
    if not continuation_token_ids:
        raise RuntimeError("Teacher-forced parity requires a non-empty continuation.")

    prompt_ids, boundary_cache, boundary_kv = _prefill_prompt_cache(
        model,
        tokenizer,
        prompt,
        args.device,
        args.layers,
        args.prefill_chunk_size,
    )
    prompt_len = prompt_ids.shape[1]

    teacher_qstates, teacher_final_cache = _teacher_forced_query_states(
        model,
        boundary_cache,
        continuation_token_ids,
        args.device,
        args.layers,
        prompt_len,
    )
    batched_qstates, batched_final_cache = _batched_continuation_query_states(
        model,
        boundary_cache,
        continuation_token_ids,
        args.device,
        args.layers,
        prompt_len,
    )

    teacher_final_kv = _extract_kv_states_from_cache(teacher_final_cache, args.layers)
    batched_final_kv = _extract_kv_states_from_cache(batched_final_cache, args.layers)

    boundary_stats: List[TensorParity] = []
    query_stats: List[TensorParity] = []
    for layer in args.layers:
        teacher_prefix_keys = teacher_final_kv[layer]["keys"][:, :prompt_len, :]
        teacher_prefix_values = teacher_final_kv[layer]["values"][:, :prompt_len, :]
        batched_prefix_keys = batched_final_kv[layer]["keys"][:, :prompt_len, :]
        batched_prefix_values = batched_final_kv[layer]["values"][:, :prompt_len, :]
        boundary_stats.append(_parity_stats(layer, "boundary_keys_teacher_vs_prompt", teacher_prefix_keys, boundary_kv[layer]["keys"]))
        boundary_stats.append(_parity_stats(layer, "boundary_values_teacher_vs_prompt", teacher_prefix_values, boundary_kv[layer]["values"]))
        boundary_stats.append(_parity_stats(layer, "boundary_keys_batched_vs_prompt", batched_prefix_keys, boundary_kv[layer]["keys"]))
        boundary_stats.append(_parity_stats(layer, "boundary_values_batched_vs_prompt", batched_prefix_values, boundary_kv[layer]["values"]))
        boundary_stats.append(_parity_stats(layer, "boundary_keys_teacher_vs_batched", teacher_prefix_keys, batched_prefix_keys))
        boundary_stats.append(_parity_stats(layer, "boundary_values_teacher_vs_batched", teacher_prefix_values, batched_prefix_values))
        query_stats.append(_parity_stats(layer, "continuation_queries_teacher_vs_batched", teacher_qstates[layer], batched_qstates[layer]))

    result = TeacherForcedParityResult(
        prompt_file=prompt_path.name,
        prefix_turns=args.prefix_turns,
        continuation_turns=args.continuation_turns,
        observed_tokens=len(continuation_token_ids),
        teacher_forced_continuation_tokens=len(continuation_token_ids),
        boundary_kv_parity=boundary_stats,
        query_parity=query_stats,
    )

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    print(f"saved parity report to {out}")
    for stat in boundary_stats + query_stats:
        print(
            f"layer={stat.layer:2d} {stat.name:<36} "
            f"max_abs={stat.max_abs:.3e} mean_abs={stat.mean_abs:.3e} "
            f"allclose={stat.allclose_atol_1e_4}"
        )


if __name__ == "__main__":
    main()
