# Phase 3C: Quotient-Aware Selection Signals

This tranche is the first direct selector-side use of the quotient-residual
lens. Its main outcome is no longer "a new selector," but the discovery that
quotient-aware information can help shortlist construction for a stronger
downstream search stack.

The key Phase 2 / 3 conclusion so far is:

- `attn_mass` sees only where attention lands
- it does **not** see whether a token's value is distinctive relative to the
  current output
- quotient-residual omission says the missing local factor is
  `w_i(q) (v_i - O(q))`

So the first conservative experiment sequence was:

1. replace pure attention-mass support ranking with a bank-aggregated
   quotient-aware omission score
2. test that shortlist-only selector directly
3. then test the same shortlist as a candidate filter for existing OMP
4. keep the rest of the pipeline fixed
5. compare against `attn_mass`, `OMP`, and `hybrid`, with and without `vfit`

This did **not** rewrite the fitting objective. It only changed which original
tokens were retained.

It also does **not** assume that local exactness solves the deployed problem.
Even if a selector improves local `E(q)`, concatenation still makes mass
fidelity matter through future competition.

---

## Initial Hypothesis

The quotient-aware selector should improve on plain `attn_mass` because it
distinguishes:

- high-attention tokens whose values are near the current output
- high- or moderate-attention tokens whose values are structurally distinctive

If the quotient story is right, this should matter most in exactly the places
where `attn_mass` repair currently fails:

- sparse or moderately rich decode-like evidence
- heads where beta-only or mass-centric support is weak

The expected outcomes are ordered by ambition:

- `quotient_omit` should beat `attn_mass` as a baseline selector
- `quotient_omit+vfit` should beat `attn_mass+vfit` if value-sensitive support
  quality is the missing ingredient
- `quotient_omit_omp` should test whether value-aware candidate pruning helps
  the existing OMP machinery without changing its objective
- if it only matches `attn_mass`, then the quotient story is not yet helping
  selection enough at the token-ranking level

---

## Current Implementation

The experiment entry point is:

- [phase3c_quotient_selector.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_quotient_selector.py)

The new selector baseline is:

- [baselines.py](/home/csmith/projects/kv_operator_matching/src/kv_operator_matching/baselines.py)

under `quotient_omission_baseline(...)`.

The shortlist + OMP follow-up in the same file is:

- `quotient_omit_omp_baseline(...)`

It supports two score modes:

- `exact_local`
  uses the exact local single-atom omission loss for normalized attention
- `proxy`
  uses a cheaper `alpha_i(q) ||v_i - O(q)||^2` proxy

The default experiment uses `exact_local`.

The current script deliberately stops short of a full E-aware OMP objective.
That remains a second-phase hypothesis, not the default next step.

---

## Metrics To Watch

Primary:

- holdout `L_true`
- mean `ΔL_true` versus `attn_mass`

Mechanism:

- holdout quotient-residual energy
- holdout quotient cancellation gain
- worst-case output-scaled quotient residual

Interpretation:

- if `quotient_omit` lowers `L_true` and improves cancellation gain, that is a
  genuine selector-side confirmation of the quotient lens
- if `quotient_omit` lowers `L_true` without improving cancellation gain, then
  the score may be helping by a different route
- if `quotient_omit+vfit` improves strongly while `quotient_omit` does not,
  then the selector is finding a better repair substrate rather than a better
  no-repair support
- if `quotient_omit_omp` improves while `quotient_omit` does not, then the
  quotient score is helping most as a candidate filter rather than as a direct
  standalone selector

---

## Null Hypothesis

The sober null is:

- most of the gain, if any, comes from better value-aware support ranking
- alternating or fully E-aware joint fitting adds little beyond the current
  `beta + C_v` pipeline once support improves

So this tranche should be treated as a selection test first, not as a mandate
to jump immediately to a larger alternating optimization scheme.

---

## Suggested First Run

Start with a narrow but substantive slice:

```bash
python experiments/qwen2_online_nz_match/phase3c_quotient_selector.py \
  --layers 4 20 \
  --budgets 0.25 0.5 \
  --collection-modes online teacher-forced-suffix repeat-prefill \
  --prompt-files near_capacity_dispatch_safe.json relational_binding_probe.json \
  --max-queries 256 \
  --max-new-tokens 32 \
  --save-json results/scratch/phase3c_quotient_selector_first_slice.json
```

If that looks promising, expand to the broader substantive prompt set before
considering full E-aware OMP or alternating joint `(beta, C_v)` refinement.

---

## First Slice Result

First non-smoke artifact:

- [phase3c_quotient_selector_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_quotient_selector_first_slice.json)

Setup:

- prompts: `near_capacity_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`
- `max_queries=256`, `max_new_tokens=32`

Initial read:

- `quotient_omit` did **not** beat plain `attn_mass` on mean holdout `L_true`
  in any of the three regimes on this first slice
  - `online`: `+0.035` versus `attn_mass`
  - `teacher-forced-suffix`: `+0.032`
  - `repeat-prefill`: `+0.197`
- despite that, `quotient_omit` did slightly improve mean quotient
  cancellation gain relative to `attn_mass` in all three regimes
  - `online`: `0.509` vs `0.492`
  - `teacher-forced-suffix`: `0.535` vs `0.518`
  - `repeat-prefill`: `0.562` vs `0.539`
- `quotient_omit+vfit` also did not beat `attn_mass+vfit`, though it stayed
  close under `repeat-prefill`
- `quotient_omit_omp` was still poor in absolute terms, but it did improve on
  raw `omp` in all three regimes
  - so quotient-aware pre-screening may still be useful as an OMP stabilizer
  - but the current shortlist+OMP path is not yet a competitive selector

So the first result narrowed the design space:

- value-aware omission scoring by itself is not yet a selector win
- it may still be capturing a real cancellation signal that is too weak at the
  current ranking level
- quotient-aware candidate filtering looks more promising as a way to improve
  OMP than as a direct replacement for `attn_mass`

That means the next follow-up should be narrow:

- inspect where `quotient_omit` improves cancellation but loses `L_true`
- test a slightly richer shortlist policy before any full E-aware OMP rewrite

That follow-up is now tracked in
[phase3c_shortlist_sweep.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_shortlist_sweep.md).

Current status:

- quotient residual is no longer only explanatory
- it is now an operationally useful shortlist signal in at least some slices
- the Phase 3C object is therefore best described as shortlist architecture,
  not a standalone selector replacement
- the broader stability slice in
  [phase3c_shortlist_sweep.md](/home/csmith/projects/kv_operator_matching/docs/phase3c_shortlist_sweep.md)
  keeps that narrower conclusion intact:
  shortlist wins survive in `online` and `repeat-prefill`, especially at
  `1.5x-2.0x` shortlist pressure, while `teacher-forced-suffix` still prefers
  plain `attn_mass`
