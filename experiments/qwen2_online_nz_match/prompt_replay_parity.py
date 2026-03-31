#!/usr/bin/env python3
"""
Parity checks for prompt-side batched prefill vs full prompt replay.

This script compares two ways of obtaining prompt-position query states and the
resulting prompt-boundary KV cache:

1. batched prefill: feed the full prompt in one forward pass
2. full prompt replay: feed the same prompt token by token from an empty cache

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
class PromptReplayParityResult:
    prompt_file: str
    prefix_turns: int
    prompt_tokens: int
    prompt_next_token_prefill: int
    prompt_next_token_replay: int
    prompt_logits_parity: TensorParity
    boundary_logits_parity: TensorParity
    kv_parity: List[TensorParity]
    query_parity: List[TensorParity]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prompt-file", default="near_capacity_dispatch_safe.json")
    p.add_argument("--prefix-turns", type=int, default=8)
    p.add_argument("--layers", nargs="+", type=int, default=[4, 20])
    p.add_argument("--save-json", default="results/scratch/prompt_replay_parity.json")
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


def _batched_prompt_states(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layers: List[int],
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    outputs, layer_inputs = _forward_with_captured_layer_inputs(
        model,
        input_ids,
        None,
        0,
        layers,
        output_attentions=False,
    )
    qstates = _compute_query_states_from_hidden_states(
        model,
        layer_inputs,
        torch.arange(prompt_len, device=device, dtype=torch.long).unsqueeze(0),
        layers,
    )
    kv_states = _extract_kv_states_from_cache(outputs.past_key_values, layers)
    return input_ids, qstates, kv_states, outputs.logits.detach().cpu()


def _replayed_prompt_states(
    model,
    input_ids: torch.Tensor,
    device: str,
    layers: List[int],
):
    prompt_len = input_ids.shape[1]
    per_layer_steps: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    per_step_logits: List[torch.Tensor] = []
    past_key_values = None
    outputs = None

    for logical_position in range(prompt_len):
        token_tensor = input_ids[:, logical_position : logical_position + 1]
        outputs, layer_inputs = _forward_with_captured_layer_inputs(
            model,
            token_tensor,
            past_key_values,
            logical_position,
            layers,
            output_attentions=False,
        )
        past_key_values = outputs.past_key_values
        per_step_logits.append(outputs.logits.detach().cpu())
        qstates = _compute_query_states_from_hidden_states(
            model,
            layer_inputs,
            torch.tensor([[logical_position]], device=device, dtype=torch.long),
            layers,
        )
        for layer in layers:
            per_layer_steps[layer].append(qstates[layer][:, 0:1, :].cpu())

    if outputs is None:
        raise RuntimeError("Prompt replay did not produce any outputs.")

    stacked_qstates = {
        layer: torch.cat(per_layer_steps[layer], dim=1) for layer in layers
    }
    kv_states = _extract_kv_states_from_cache(past_key_values, layers)
    stacked_logits = torch.cat(per_step_logits, dim=1)
    return stacked_qstates, kv_states, stacked_logits


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.device)

    smoke_data_path = REPO_ROOT.parent / "kv_compaction_experiment" / "data" / "smoke_test"
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = smoke_data_path / args.prompt_file
    prompt, _continuation_text = load_prompt_segments_from_file(
        prompt_path,
        prefix_turns=args.prefix_turns,
        continuation_turns=None,
    )

    input_ids, prefill_qstates, prefill_kv, prefill_logits = _batched_prompt_states(
        model,
        tokenizer,
        prompt,
        args.device,
        args.layers,
    )
    replay_qstates, replay_kv, replay_logits = _replayed_prompt_states(
        model,
        input_ids,
        args.device,
        args.layers,
    )

    kv_stats: List[TensorParity] = []
    query_stats: List[TensorParity] = []
    for layer in args.layers:
        kv_stats.append(
            _parity_stats(layer, "prompt_keys_prefill_vs_replay", prefill_kv[layer]["keys"], replay_kv[layer]["keys"])
        )
        kv_stats.append(
            _parity_stats(layer, "prompt_values_prefill_vs_replay", prefill_kv[layer]["values"], replay_kv[layer]["values"])
        )
        query_stats.append(
            _parity_stats(layer, "prompt_queries_prefill_vs_replay", prefill_qstates[layer], replay_qstates[layer])
        )

    prompt_logits_parity = _parity_stats(-1, "prompt_logits_prefill_vs_replay", prefill_logits, replay_logits)
    boundary_logits_parity = _parity_stats(
        -1,
        "boundary_logits_prefill_vs_replay",
        prefill_logits[:, -1:, :],
        replay_logits[:, -1:, :],
    )

    result = PromptReplayParityResult(
        prompt_file=prompt_path.name,
        prefix_turns=args.prefix_turns,
        prompt_tokens=int(input_ids.shape[1]),
        prompt_next_token_prefill=int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item()),
        prompt_next_token_replay=int(torch.argmax(replay_logits[:, -1, :], dim=-1).item()),
        prompt_logits_parity=prompt_logits_parity,
        boundary_logits_parity=boundary_logits_parity,
        kv_parity=kv_stats,
        query_parity=query_stats,
    )

    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    print(f"saved parity report to {out}")
    print(
        f"prefill_next_token={result.prompt_next_token_prefill} "
        f"replay_next_token={result.prompt_next_token_replay}"
    )
    print(
        f"layer={prompt_logits_parity.layer:2d} {prompt_logits_parity.name:<36} "
        f"max_abs={prompt_logits_parity.max_abs:.3e} mean_abs={prompt_logits_parity.mean_abs:.3e} "
        f"allclose={prompt_logits_parity.allclose_atol_1e_4}"
    )
    print(
        f"layer={boundary_logits_parity.layer:2d} {boundary_logits_parity.name:<36} "
        f"max_abs={boundary_logits_parity.max_abs:.3e} mean_abs={boundary_logits_parity.mean_abs:.3e} "
        f"allclose={boundary_logits_parity.allclose_atol_1e_4}"
    )
    for stat in kv_stats + query_stats:
        print(
            f"layer={stat.layer:2d} {stat.name:<36} "
            f"max_abs={stat.max_abs:.3e} mean_abs={stat.mean_abs:.3e} "
            f"allclose={stat.allclose_atol_1e_4}"
        )


if __name__ == "__main__":
    main()
