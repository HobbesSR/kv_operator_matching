# Phase 2 Evidence Collection Note

This note records what was learned from `kv_compaction_experiment`, how those
lessons map onto this repo, and the concrete Phase 2 task list that follows.

---

## Current Regimes

The experiment harness now has four distinct query-collection regimes:

- `prefill`: encode-time queries over a known prompt prefix
- `repeat-prefill`: offline control using `prompt + "Repeat it." + prompt`
- `teacher-forced`: decode-time queries on a fixed known continuation
- `online`: decode-time queries on the model's own continuation

The intended compacted object for `teacher-forced` and `online` is the
**prompt-boundary cache**, not the final cache after continuation tokens have
been appended.

---

## Lessons Imported From `kv_compaction_experiment`

### 1. Separate correctness from throughput

The old repo used explicit parity checks instead of trusting the online path by
inspection. We should do the same. A collector that "runs" is not enough.

### 2. Teacher-forced decode is a first-class control

The old repo repeatedly used teacher-forced harvests as the stable dense
control. That matches our current need: decode-like supervision without
sampling noise.

### 3. Full eager outputs are expensive

The old repo found that `output_hidden_states=True` /
`output_attentions=True` on every live step imposed a large evidence tax.
Pushing extraction closer to the attention forward path reduced cost, but did
not eliminate Python-side bookkeeping overhead.

### 4. Evidence-population mismatch is a real scientific risk

The old repo explicitly found cases where retained evidence was dominated by
one population of tokens while selected survivors came from another. That means
we must record evidence coverage and source mix, not just final losses.

### 5. Normalization is diagnostic, not magic

Prompt replay plus normalization made evidence collection parity more honest in
the old repo, but it did not automatically fix the underlying failure mode.
Coverage-aware accounting is therefore necessary, but not sufficient.

---

## Pitfalls To Avoid Here

- Do not evaluate online queries against a future cache state.
- Do not silently replace EOS with some other token to "keep the collector
  alive."
- Do not compare raw holdout losses across different collection regimes as if
  the holdout banks were interchangeable.
- Do not hide differences in supervision density. Report observed positions,
  retained rows, and retained-opportunity fraction.
- Do not assume the collector is correct just because the downstream metrics
  look plausible.

---

## Tonight's Task List

- [x] Fix causal leakage in the `online` collector by holding the compacted
      object at the prompt boundary.
- [x] Remove off-policy EOS suppression from the `online` collector.
- [x] Add `repeat-prefill` as an explicit control path.
- [x] Add `teacher-forced` decode as a middle collection regime.
- [x] Add opportunity-normalized evidence accounting to regime-comparison
      artifacts.
- [x] Add a parity harness for teacher-forced decode vs batched continuation
      prefill.
- [x] Run a three-regime comparison (`online`, `teacher-forced`,
      `repeat-prefill`) on a prompt/layer/budget matrix.
- [x] Summarize where the program now stands after the corrected runs.

Artifacts from tonight:

- [teacher_forced_parity.json](/home/csmith/projects/kv_operator_matching/results/teacher_forced_parity.json)
- [teacher_forced_sanity.json](/home/csmith/projects/kv_operator_matching/results/teacher_forced_sanity.json)
- [collection_mode_comparison_teacher_forced_small.json](/home/csmith/projects/kv_operator_matching/results/collection_mode_comparison_teacher_forced_small.json)
- [collection_mode_comparison_with_teacher_forced.json](/home/csmith/projects/kv_operator_matching/results/collection_mode_comparison_with_teacher_forced.json)

---

## Next Task List After Tonight

- [ ] Promote the parity harness to a routine regression check.
- [ ] Investigate the residual teacher-forced query parity gap:
      boundary KV parity is exact, but continuation queries are only
      approximately equal between stepwise decode and batched continuation
      prefill.
- [ ] Add source-mix reporting once prompt-side and decode-side evidence are
      mixed within the same artifact.
- [ ] Expand the three-regime comparison over more prompts and longer
      continuations.
- [ ] If online collection becomes a bottleneck, move evidence extraction
      closer to the attention forward path instead of relying on generic full
      output capture.
- [ ] Revisit support-quality gating only after the collection regime is
      stable and parity-checked.

---

## Where We Stand Tonight

### Correctness

- The `online` collector is now causal and on-policy.
- The `teacher-forced` collector keeps the compacted object fixed at the
  prompt boundary and uses a known continuation one token at a time.
- The parity harness confirms:
  - **exact boundary KV parity** between prompt-only prefill, teacher-forced
    decode, and batched continuation prefill
  - **non-exact continuation query parity** between teacher-forced stepping and
    batched continuation prefill, with small mean error but noticeable max
    deviations on deeper layers

### Evidence Accounting

- Regime-comparison artifacts now record:
  - observed positions
  - raw query vectors per bank
  - retained query vectors per bank
  - retained-position equivalent
  - retained-opportunity fraction
  - skipped prompt/mode pairs
- The main single-run experiment artifact now also records collection metadata
  and opportunity-normalized retention.

### Three-Regime Read

With `online`, `teacher-forced`, and `repeat-prefill` all present:

- `recency+refit` is still harmful in every regime:
  - `online`: mean holdout `ΔL_true = +2.46`, improved `0/16`
  - `teacher-forced`: `+3.00`, improved `0/16`
  - `repeat-prefill`: `+5.43`, improved `0/24`
- `recency+vfit` is consistently helpful, strongest under `repeat-prefill`,
  next under `teacher-forced`, weakest under `online`:
  - `online`: `-1.42`, improved `14/16`
  - `teacher-forced`: `-3.13`, improved `16/16`
  - `repeat-prefill`: `-4.12`, improved `24/24`
- `attn_mass+refit` is near break-even to mildly harmful; `teacher-forced`
  looks a bit better than `online` on the current matrix:
  - `online`: `+0.35`, improved `1/16`
  - `teacher-forced`: `+0.23`, improved `4/16`
  - `repeat-prefill`: `+0.29`, improved `6/24`
- `attn_mass+vfit` is:
  - harmful under `online`
  - still harmful under `teacher-forced`
  - helpful on average under `repeat-prefill`
  - specifically:
    - `online`: `+1.00`, improved `0/16`
    - `teacher-forced`: `+0.72`, improved `0/16`
    - `repeat-prefill`: `-0.35`, improved `9/24`
- `attn_mass+phase1b` remains weak across all three regimes:
  - `online`: `+1.25`, improved `0/16`
  - `teacher-forced`: `+0.98`, improved `0/16`
  - `repeat-prefill`: `+0.25`, improved `5/24`

The current program state is therefore:

- `teacher-forced` is the right middle control and behaves differently from
  both `online` and `repeat-prefill`
- opportunity-normalized supervision density matters
- the remaining uncertainty is no longer "do we have the regimes?" but
  "which methods survive once decode-like supervision is both correct and
  dense enough"
