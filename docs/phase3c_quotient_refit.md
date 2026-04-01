# Phase 3C: Fixed-Support Quotient-Aware Refit

The shortlist tranche answered the selection-side question:

- quotient-aware information can improve shortlist construction
- but mostly as a search prior rather than as a standalone selector

This tranche isolates the other half of the Phase 3C decomposition:

- keep support fixed
- compare ordinary anchored `vfit`
- against quotient-aware fixed-support `qvfit`

So the question is no longer "does quotient help choose supports?" but:

- does quotient-aware refitting improve bank-faithful reconstruction even when
  support choice is unchanged?

---

## Objective

For fixed support keys and fixed `beta`, ordinary `vfit` solves:

$$
\min_V \sum_q w_q \|\hat O(q; V) - O(q)\|^2
$$

using the compact normalized attention weights as the design matrix.

The quotient-aware refit instead fixes the same support and solves the exact
local fixed-support quotient-residual problem:

$$
\min_V \sum_q w_q \|E(q; V)\|^2
$$

with

$$
E(q; V) = \hat N(q; V) - O(q)\hat Z(q).
$$

Because support keys and `beta` are fixed, `\hat Z(q)` is fixed, so this is
still a linear least-squares solve in `V`.

Implementation note:

- `vfit` uses normalized compact attention weights as the design matrix
- `qvfit` uses the unnormalized compact numerator weights under a shared
  stable query shift and targets `\hat Z(q) O(q)`

This is still a local bank objective, not a concatenation-aware deployed
objective.

---

## Experiment

Entry point:

- [phase3c_quotient_refit.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_quotient_refit.py)

Refit implementations:

- [value_fit.py](/home/csmith/projects/kv_operator_matching/src/kv_operator_matching/value_fit.py)

The first substantive slice compares:

- `attn_mass+vfit` vs `attn_mass+qvfit`
- `omp+vfit` vs `omp+qvfit`
- `hybrid+vfit` vs `hybrid+qvfit`
- `quotient_omit_omp+vfit` vs `quotient_omit_omp+qvfit`

So this is the first direct test of the "ordinary selector + quotient-aware
refit" and "quotient-aware selector + quotient-aware refit" cells in the 2x2
decomposition.

---

## First Run

```bash
python experiments/qwen2_online_nz_match/phase3c_quotient_refit.py \
  --layers 4 20 \
  --budgets 0.25 0.5 \
  --collection-modes online teacher-forced-suffix repeat-prefill \
  --prompt-files near_capacity_dispatch_safe.json relational_binding_probe.json \
  --max-queries 256 \
  --max-new-tokens 32 \
  --save-json results/scratch/phase3c_quotient_refit_first_slice.json
```

Artifact:

- [phase3c_quotient_refit_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_quotient_refit_first_slice.json)

---

## First Result

The fixed-support quotient-aware refit is a real lever, but only for some
support families.

Strongest result:

- `attn_mass+qvfit` beat `attn_mass+vfit` in all three regimes
  - `online`: `-0.20`
  - `teacher-forced-suffix`: `-0.27`
  - `repeat-prefill`: `-0.29`

Mixed or negative results elsewhere:

- `hybrid+qvfit` was worse than `hybrid+vfit`
  - `online`: `+0.66`
  - `teacher-forced-suffix`: `+0.55`
  - `repeat-prefill`: effectively flat at `+0.00`
- `omp+qvfit` was substantially worse than `omp+vfit` under `online` and
  `teacher-forced-suffix`, and only slightly better under `repeat-prefill`
  (`-0.03`)
- `quotient_omit_omp+qvfit` also lost under `online` / `teacher-forced-suffix`
  and only modestly helped under `repeat-prefill` (`-0.08`)

So the first refit-side answer is:

- yes, quotient-aware refitting can matter even with support held fixed
- no, it is not a universal upgrade over ordinary anchored `vfit`

That means the support/refit decomposition is now empirically real on both
sides:

- shortlist-side quotient information helps some support searches
- refit-side quotient information helps some fixed supports

But the leverage is clearly support-sensitive.

---

## Interpretation

The current evidence suggests:

1. `qvfit` is strongest for coarse mass-centric supports.
   `attn_mass` appears to benefit from the extra quotient weighting in a way
   ordinary `vfit` was missing.

2. `qvfit` can be harmful for already aggressive or geometry-rich supports.
   `OMP` and `hybrid` likely already concentrate support in directions where
   the unnormalized quotient objective overweights unstable high-mass parts of
   the bank.

3. The best current "ordinary selector + quotient-aware refit" cell is much
   stronger than the first "quotient-aware selector + quotient-aware refit"
   cell.
   So the immediate quotient-refit win is not a fully aligned quotient stack.
   It is a targeted rescue for weaker support families.

Current takeaway:

- quotient-aware refit is now operationally relevant
- but it is support-conditioned, not a new default refit rule

---

## Next Question

The next refit-side question is not "does qvfit work at all?" That is now
answered.

It is:

- what support geometry predicts whether `qvfit` helps or destabilizes?

That suggests the next mechanism tranche should inspect:

- support design conditioning before `vfit` versus `qvfit`
- whether `qvfit` gains are concentrated in higher-`Z` query regions
- whether a blended or regularized quotient-aware refit can keep the
  `attn_mass` win without damaging `OMP` / `hybrid`
