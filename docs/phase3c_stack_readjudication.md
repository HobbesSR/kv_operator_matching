# Phase 3C: Stack Re-Adjudication on the Canonical Broad Surface

The quotient program had already produced two separate outcomes:

- the best current refit policy is `qfit_diag`
- the best previously established broad support story is still the Phase 3A
  hybrid selector

That left one unresolved question:

> what is the best tested end-to-end stack on the canonical broad Phase 3
> surface?

This tranche re-adjudicates that question directly instead of treating the
refit and support branches as separate leaderboards.

---

## Winner Criterion

Use exactly one primary criterion:

- mean holdout `L_true` on the canonical broad `q256/t32` Phase 3 surface

Regime breakdowns are secondary.

---

## Surface

This reuses the original broad Phase 3A-style surface:

- prompts:
  - `near_capacity_dispatch.json`
  - `near_capacity_dispatch_safe.json`
  - `near_capacity_network_dispatch_safe.json`
  - `relational_binding_probe.json`
- layers: `4`, `12`, `20`, `28`
- budgets: `25%`, `50%`
- regimes:
  - `online`
  - `teacher-forced-suffix`
  - `repeat-prefill`
- query-bank density: `q256`

Artifact:

- [phase3c_stack_readjudication_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_stack_readjudication_q256_t32.json)

---

## Methods Compared

Incumbent broad support baseline:

- `hybrid+vfit`

Refit-updated incumbent:

- `hybrid+qfit_diag`

Other major stack candidates:

- `attn_mass+vfit`
- `attn_mass+qfit_diag`
- `omp+vfit`
- `omp+qfit_diag`
- `quotient_omit_omp+qfit_diag`
- `rank_blend_omp+qfit_diag`
- `two_stage_gate_omp+qfit_diag`

So this is a stack re-adjudication, not a refit-only or shortlist-only slice.

---

## Main Result

On the canonical broad surface, the best tested overall stack is now:

- `attn_mass+qfit_diag`

Overall mean holdout `L_true`:

- `attn_mass+qfit_diag`: `4.0124`
- `attn_mass+vfit`: `4.3411`
- `hybrid+qfit_diag`: `5.9990`
- `hybrid+vfit`: `6.0111`
- `quotient_omit_omp+qfit_diag`: `6.3444`
- `two_stage_gate_omp+qfit_diag`: `6.3444`
- `rank_blend_omp+qfit_diag`: `6.3473`
- `omp+qfit_diag`: `7.1604`
- `omp+vfit`: `7.1630`

Against the incumbent `hybrid+vfit`, the new winner improves mean holdout
`L_true` by:

- `-1.9987`

That is not a tie-level change. It is a decisive reorder of the broad stack
leaderboard.

---

## Regime Breakdown

`attn_mass+qfit_diag` also wins each regime individually on this surface:

- `online`: `1.6508`
- `teacher-forced-suffix`: `1.9938`
- `repeat-prefill`: `8.3927`

Compared with the previous incumbent:

- `hybrid+vfit`
  - `online`: `4.0375`
  - `teacher-forced-suffix`: `4.4066`
  - `repeat-prefill`: `9.5892`

So this is not just a mixed-surface average artifact. The same stack wins
under all three evidence regimes on the broad surface.

---

## What Changed

The broad re-adjudication says something sharper than
"`qfit_diag` is the best refit policy."

It says:

- the measured `qfit` controller changes the support ranking itself
- once that controller is available, the old support hierarchy no longer holds
- the simplest mass-centric support family becomes the best tested broad stack

More concretely:

- `hybrid+qfit_diag` only slightly improves over `hybrid+vfit`
- OMP-family supports remain clearly weaker overall
- the newer quotient-shortlist support families do not yet beat the old hybrid
  line on this broad surface
- but `attn_mass` paired with the measured `qfit` controller now beats all of
  them

So the current best explanation is:

- support simplicity plus the right adaptive refit can beat a more elaborate
  support family whose main advantage was previously realized under `vfit`

---

## Interpretation

This does **not** mean the selector branch was wasted.

The selector branch still established:

- quotient-aware shortlist information is operationally real
- shortlist architecture can improve downstream search opportunity

But the current broad-stack winner is now on the refit side:

- `attn_mass+qfit_diag`

So the project's current best tested stack is no longer "best support family
under anchored `vfit`." It is:

- a simple support family
- plus a measured row-metric controller over the `qfit` family

That is a real strategic shift.

---

## Current Status

The repo's current broad-stack status is now:

- best refit policy baseline: `qfit_diag`
- best previously established broad support baseline: Phase 3A hybrid
- best tested overall stack on the canonical broad surface:
  `attn_mass+qfit_diag`

The next question is no longer whether `qfit_diag` merely helps.
It does.

The next question is why the measured controller reshuffles the support
hierarchy this strongly, and whether that reordering survives denser or more
stressful surfaces.
