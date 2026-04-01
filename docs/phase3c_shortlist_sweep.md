# Phase 3C Follow-Up: Structural Shortlist Sweep

The first quotient-aware selector tranche narrowed the design space:

- pure `quotient_omit` was not a better standalone selector than `attn_mass`
- quotient-aware pre-screening did appear to improve OMP consistently
- so the next question is about shortlist construction, not end-to-end
  quotient optimization

This follow-up fixes the downstream solver and varies only the shortlist
policy. That is now the preferred Phase 3C framing:

- quotient-aware shortlist architecture
- fixed downstream OMP + `vfit`
- no claim yet that quotient should replace the rest of the support-search
  stack

---

## Primary Question

Does quotient-aware shortlist construction improve downstream support quality
when the downstream solver is held fixed?

Secondary question:

- is quotient-aware information mainly useful at tight shortlist budgets, where
  search-space triage matters most?

---

## Policies

The first structural shortlist sweep compares:

- `attn_mass`
- `quotient_omit`
- `rank_blend`
- `two_stage_gate`

where:

- `rank_blend` is an equal-weight rank-sum of attention-mass and quotient ranks
- `two_stage_gate` keeps a larger pool by attention mass, then prunes inside it
  by quotient score

The downstream solver is fixed:

- OMP on the shortlist
- then anchored `vfit`

This keeps the experiment interpretable. No alpha sweep is needed for the first
pass.

---

## Diagnostics

Beyond holdout `L_true`, this sweep records:

- quotient cancellation gain
- oracle support overlap
- oracle support retention inside the shortlist
- conformist-filter diagnostics

The conformist diagnostic tracks whether quotient-heavy policies preferentially
exclude tokens that have:

- high attention mass
- low value deviation from the local output

That is the specific mechanism the quotient story predicts.

---

## First Run

```bash
python experiments/qwen2_online_nz_match/phase3c_shortlist_sweep.py \
  --layers 4 20 \
  --budgets 0.25 0.5 \
  --collection-modes online teacher-forced-suffix repeat-prefill \
  --prompt-files near_capacity_dispatch_safe.json relational_binding_probe.json \
  --max-queries 256 \
  --max-new-tokens 32 \
  --save-json results/scratch/phase3c_shortlist_sweep_first_slice.json
```

The intended read order is:

1. performance vs shortlist multiplier for each policy
2. shortlist recall / overlap diagnostics
3. conformist-filter diagnostics

If no structural policy beats plain `attn_mass` shortlist construction, the
quotient signal is probably more explanatory than operational at this stage.

---

## First Slice Result

First artifact:

- [phase3c_shortlist_sweep_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_shortlist_sweep_first_slice.json)

Setup matched the first Phase 3C selector slice:

- prompts: `near_capacity_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`
- shortlist multipliers: `1.5x`, `2x`, `3x`, `4x`

Initial read:

- there **is** operational shortlist signal, but it is not universal
- the cleanest wins were:
  - `online`, `2.0x`: `rank_blend` beat `attn_mass` by `-0.20`
  - `repeat-prefill`, `1.5x`: `quotient_omit` beat `attn_mass` by `-0.20`
  - `repeat-prefill`, `4.0x`: `rank_blend` beat `attn_mass` by `-0.16`
- `teacher-forced-suffix` was much less friendly:
  - no policy produced a clear win at tight shortlist sizes
  - the best result was effectively tie-level at `3.0x`

So the shortlist result upgrades the project status:

- quotient-aware information is not just explanatory
- it can help as shortlist construction for OMP + `vfit`
- but the gain is regime- and shortlist-size-sensitive rather than broadly
  dominant

Two additional reads matter:

1. Oracle support recall / overlap often improved under quotient-heavy
   shortlist policies, especially at `2x-3x` multipliers.
   That supports the "useful search prior" interpretation.

2. The conformist-filter diagnostic did **not** move much.
   Quotient-heavy policies did not strongly reduce conformist retention on this
   slice.
   So the operational gain is real enough to matter, but the simple
   "filters conformists" mechanism is not yet strongly confirmed here.

Current interpretation:

- shortlist construction is the right place to keep using quotient information
- equal-rank blend and quotient-only shortlist both have live pockets
- two-stage gating did not clearly beat pure quotient shortlist on this slice
- the best next follow-up is to inspect the regime asymmetry, especially why
  `teacher-forced-suffix` remains resistant while `online` and
  `repeat-prefill` show pockets of lift

The key question is now:

- where does quotient shortlist value come from
- and how stable is that effect across broader slices

---

## Stability Slice Result

Broader artifact:

- [phase3c_shortlist_sweep_stability_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_shortlist_sweep_stability_slice.json)

Broader setup:

- prompts: `near_capacity_dispatch`, `near_capacity_dispatch_safe`,
  `near_capacity_network_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`
- shortlist multipliers: `1.5x`, `2x`

Command:

```bash
python experiments/qwen2_online_nz_match/phase3c_shortlist_sweep.py \
  --layers 4 20 \
  --budgets 0.25 0.5 \
  --collection-modes online teacher-forced-suffix repeat-prefill \
  --prompt-files near_capacity_dispatch.json near_capacity_dispatch_safe.json \
    near_capacity_network_dispatch_safe.json relational_binding_probe.json \
  --max-queries 256 \
  --max-new-tokens 32 \
  --shortlist-multipliers 1.5 2.0 \
  --save-json results/scratch/phase3c_shortlist_sweep_stability_slice.json
```

Stability read:

- the shortlist signal survives the broader slice, but it stays narrow rather
  than universal
- the best pockets are still under tight shortlist pressure
  - `online`, `1.5x`: `quotient_omit` beat `attn_mass` by `-0.03`
  - `online`, `2.0x`: `rank_blend` beat `attn_mass` by `-0.15`
  - `repeat-prefill`, `1.5x`: `quotient_omit` beat `attn_mass` by `-0.23`
  - `repeat-prefill`, `2.0x`: `quotient_omit` still edged `attn_mass` by
    `-0.01`
- `teacher-forced-suffix` remains the clear exception
  - `attn_mass` was still best at both `1.5x` and `2.0x`
  - quotient-heavy shortlist policies improved some overlap diagnostics there,
    but not final holdout `L_true`

Two additional points matter:

1. Oracle-support overlap and oracle-in-shortlist rates often remained strong
   or improved under quotient-heavy policies in the regimes where they helped.
   That keeps the interpretation anchored on candidate-pool quality rather than
   on a lucky score artifact.

2. The conformist-filter diagnostic still moved very little on the broader
   slice.
   So the operational benefit is real enough to matter, but the simple
   "quotient wins by dropping high-mass conformists" story is still too weak.

Current interpretation:

- quotient-aware information is now operationally useful for shortlist
  construction
- the benefit is strongest when shortlist pressure is real, especially in
  `online` and `repeat-prefill`
- the mechanism currently looks more like improved downstream search
  opportunity than a standalone selector principle
- the next Phase 3C question is therefore stability and mechanism:
  why `teacher-forced-suffix` stays resistant, and what shortlist contents
  differ materially between `attn_mass` and quotient-heavy policies
