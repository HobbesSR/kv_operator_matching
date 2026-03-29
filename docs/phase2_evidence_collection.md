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

- [teacher_forced_parity.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/teacher_forced_parity.json)
- [teacher_forced_sanity.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/teacher_forced_sanity.json)
- [collection_mode_comparison_teacher_forced_small.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_teacher_forced_small.json)
- [collection_mode_comparison_with_teacher_forced.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_with_teacher_forced.json)
- [collection_mode_comparison_substantive_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_substantive_q256_t32.json)
- [collection_mode_comparison_substantive_q512_t64.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_substantive_q512_t64.json)
- [forensic_support_geometry_q512_t64.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/forensic_support_geometry_q512_t64.json)
- [collection_mode_comparison_with_omp_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_with_omp_q256_t32.json)
- [collection_mode_comparison_with_omp_q512_t64_l420.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/collection_mode_comparison_with_omp_q512_t64_l420.json)
- [forensic_support_geometry_with_omp_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase2/forensic_support_geometry_with_omp_q256_t32.json)

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

### Broader Substantive-Prompt Sweep

On the broader sweep over 4 substantive prompts, 4 layers, 2 budgets, and both
KV heads (`64` holdout cells per regime-method pairing):

- With `max_queries=256`, `max_new_tokens=32`:
  - `recency+vfit` is the strongest result in every regime:
    - `online`: mean holdout `ΔL_true = -2.76`, improved `59/64`
    - `teacher-forced`: `-5.90`, improved `64/64`
    - `repeat-prefill`: `-5.16`, improved `64/64`
  - `recency+phase1b` becomes mildly to strongly positive:
    - `online`: `-0.54`, improved `33/64`
    - `teacher-forced`: `-3.10`, improved `46/64`
    - `repeat-prefill`: `-0.08`, improved `34/64`
  - `attn_mass+refit` is still weak:
    - `online`: `+0.17`
    - `teacher-forced`: `+0.09`
    - `repeat-prefill`: `+0.51`
  - `attn_mass+vfit` still needs denser or different supervision:
    - `online`: `+1.03`
    - `teacher-forced`: `+0.83`
    - `repeat-prefill`: `-0.16`

- With denser evidence (`max_queries=512`, `max_new_tokens=64`):
  - `recency+vfit` stays dominant and improves further under decode-like
    regimes:
    - `online`: `-3.10`, improved `61/64`
    - `teacher-forced`: `-6.16`, improved `64/64`
    - `repeat-prefill`: `-5.02`, improved `64/64`
  - `recency+phase1b` stays positive under `online` and `teacher-forced`:
    - `online`: `-0.59`
    - `teacher-forced`: `-3.18`
    - `repeat-prefill`: `+0.35`
  - `attn_mass+vfit` improves slightly under `online` and `teacher-forced`,
    but remains harmful there:
    - `online`: `+0.83`
    - `teacher-forced`: `+0.72`
    - `repeat-prefill`: `-0.35`
  - `attn_mass+refit` gets worse rather than better with more evidence.

The current broad Phase 2 read is therefore:

- `recency+vfit` is now the main positive signal, not a small curiosity.
- `teacher-forced` is not merely a control; for the recency value-repair
  family it is the strongest evidence regime so far.
- `attn_mass` local repair still looks supervision-limited or mismatched even
  after broadening the matrix.
- More evidence does not rescue the `beta` stage in the attention-mass path.

Layer-pattern note from the denser sweep:

- `recency+vfit` is broadly positive on essentially every layer:
  - `online`: improved `16/16` cells on layers 4, 12, and 28, and `13/16` on
    layer 20
  - `teacher-forced`: improved `16/16` on all four tested layers
- `attn_mass` repairs degrade with depth rather than stabilizing:
  - `attn_mass+vfit` remains harmful on every tested `online` layer and on
    every tested `teacher-forced` layer, with the worst mean deltas at layer 28
  - `attn_mass+phase1b` is uniformly harmful across all tested `online` layers
    and nearly so under `teacher-forced`

### Support-Geometry Forensics

The comparative geometry pass on the dense substantive setup supports the
"repair substrate" hypothesis directly.

- `recency` supports are much more local and coherent:
  - mean support age fraction is about `0.19` in every regime
  - support span fraction is about `0.37`
  - adjacency fraction is exactly `1.0`
- `attn_mass` supports are older and nearly full-span:
  - mean support age fraction is about `0.36` online, `0.42` teacher-forced,
    and `0.47` repeat-prefill
  - support span fraction is effectively `1.0`
  - adjacency fraction is only about `0.48-0.61`

- `recency` also yields much healthier design geometry under decode-like
  regimes:
  - mean stable rank is `7.11` online and `6.81` teacher-forced
  - for `attn_mass`, it is only `4.21` online and `3.96` teacher-forced

- The value updates themselves are not smaller on `recency`; if anything they
  are slightly larger on average. The difference is therefore not "recency
  changes less." The difference is that the same anchored update lands on a
  more coherent support.

- Post-repair error concentration is lower on `recency`:
  - holdout top-5 error share is `0.136` online and `0.105` teacher-forced
  - for `attn_mass`, it is `0.199` online and `0.192` teacher-forced

The current best explanation is therefore:

- `recency` is a better substrate for anchored local value repair under
  decode-like evidence
- `attn_mass` may be a good replay selector, but it looks too dispersed and
  heterogeneous to serve as a stable local repair basis in the same regime

### OMP Control

We added a local per-head OMP support selector as a closer control for the
paper's AM-OMP recipe. In this repo, the closest current analogue is:

- `omp`: OMP-selected support with fitted beta coefficients
- `omp+vfit`: OMP support+beta followed by anchored value repair

Results on the substantive `q256 / t32` sweep:

- `omp+vfit` is a real positive relative to its own baseline in every regime:
  - `online`: mean holdout `ΔL_true = -3.35`, improved `54/64`
  - `teacher-forced`: `-4.17`, improved `59/64`
  - `repeat-prefill`: `-2.93`, improved `58/64`

But the absolute ranking is more nuanced:

- `omp+vfit` beats `recency+vfit` under:
  - `teacher-forced`: `45/64` cells, mean absolute `L_true` difference `-1.10`
  - `repeat-prefill`: `55/64` cells, mean difference `-1.85`
- `omp+vfit` loses to `recency+vfit` under:
  - `online`: `27/64` cells, mean difference `+0.53`

At the same time, the OMP family still trails the attention-mass family in
absolute `L_true`:

- `omp` baseline is much worse than `attn_mass` baseline in every regime.
- `omp+vfit` still loses to `attn_mass+vfit` in absolute `L_true` on the
  current matrices, even though `attn_mass+vfit` often harms relative to the
  stronger `attn_mass` baseline.

The reduced dense OMP check (`layers = 4, 20`, `q512 / t64`) tells the same
story:

- `omp+vfit` stays strongly positive relative to `omp` in every regime.
- It still beats `recency+vfit` under `teacher-forced` and `repeat-prefill`,
  but not under `online`.
- It still does not beat `attn_mass+vfit` in absolute `L_true`.

So the OMP control changes the picture, but not all the way:

- OMP helps enough to narrow the gap with the recency family and even win in
  richer evidence regimes.
- OMP does not rescue the whole attention-matching direction on the current
  local mechanistic surface, because the absolute `attn_mass` baselines remain
  much stronger and the online regime still favors `recency+vfit`.

### OMP vs Recency Geometry

The follow-up geometry pass clarifies *why* OMP beats recency in richer regimes
but still loses online.

- OMP does **not** look like the more stable local repair substrate:
  - its supports are older and nearly full-span
  - its adjacency fraction is much lower than recency
  - its design stable rank is lower than recency in all three regimes
  - its low-singular-direction update share is higher than recency in all
    three regimes

On the broad `q256 / t32` surface:

- `online`
  - `recency` baseline mean holdout `L_true = 7.74`
  - `omp` baseline mean holdout `L_true = 8.87`
  - `recency+vfit` ends at `4.98`
  - `omp+vfit` ends at `5.52`

- `teacher-forced`
  - `recency` baseline mean holdout `L_true = 12.64`
  - `omp` baseline mean holdout `L_true = 9.80`
  - `recency+vfit` ends at `6.74`
  - `omp+vfit` ends at `5.63`

- `repeat-prefill`
  - `recency` baseline mean holdout `L_true = 16.25`
  - `omp` baseline mean holdout `L_true = 12.85`
  - `recency+vfit` ends at `11.51`
  - `omp+vfit` ends at `9.75`

So the richer-regime OMP win is mostly a **baseline-quality win**, not a
"better-conditioned repair substrate" win:

- under `teacher-forced` and `repeat-prefill`, OMP starts from a much better
  baseline than recency, and value repair preserves that advantage
- under true `online`, recency starts from the better baseline and keeps the
  lead even though OMP's value-repair delta is slightly larger

The current best explanation is therefore:

- `recency` remains the more coherent local value-repair substrate
- `OMP` becomes competitive or better only when the richer evidence regime
  makes its stronger baseline support quality matter more than its weaker local
  geometry
