# Phase 3C: Gated And Tempered QVFit Policy

The compatibility tranche produced a concrete control variable:

- `zhat_over_zref_cv`

That suggested a real refit-side policy test rather than another abstract
mechanism note.

This tranche compares four fixed-support refit policies:

- plain `vfit`
- raw `qvfit`
- hard-gated `qvfit`
- tempered `qvfit`

The gate uses the compatibility statistic directly:

- use `qvfit` only if `zhat_over_zref_cv <= 0.25`

The tempered variant keeps the quotient-aware solve but reduces row-scaling
severity by using a power-tempered row scale (`gamma = 0.5`) instead of the
full quotient scale.

---

## Policy Logic

This is not a free blend-weight story.

The hard gate is a structural control law:

- if quotient-induced row-scaling dispersion is low, use `qvfit`
- otherwise fall back to ordinary `vfit`

The tempered solve is a softer control:

- keep quotient-aware row scaling
- but compress its magnitude so the fit is less hostage-like to a few rows

So the tranche asks a narrow question:

- can control of quotient row scaling turn `qvfit` from a risky objective into
  a better policy?

---

## Experiment

Entry point:

- [phase3c_qvfit_policy.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_qvfit_policy.py)

The row-scaling controls are implemented in:

- [value_fit.py](/home/csmith/projects/kv_operator_matching/src/kv_operator_matching/value_fit.py)

---

## First Slice

Artifact:

- [phase3c_qvfit_policy_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_policy_first_slice.json)

Setup:

- prompts: `near_capacity_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`

Overall mean holdout `L_true`:

- `vfit`: `5.398`
- raw `qvfit`: `5.866`
- hard-gated `qvfit`: `5.329`
- tempered `qvfit`: `5.617`

So on the first slice:

- hard-gated `qvfit` beat plain `vfit`
- raw `qvfit` remained much worse
- tempered `qvfit` recovered part of the damage but did not beat the gate

Interpretation:

- the gate preserves the real `attn_mass` wins
- it removes most of the catastrophic `OMP`-family failures
- it is already a better refit policy than always using either extreme

---

## Broader Validation

Artifact:

- [phase3c_qvfit_policy_stability_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_policy_stability_slice.json)

Setup matched the broader stability surface:

- prompts: `near_capacity_dispatch`, `near_capacity_dispatch_safe`,
  `near_capacity_network_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`

Overall mean holdout `L_true` on the broader slice:

- `vfit`: `5.275`
- raw `qvfit`: `5.698`
- hard-gated `qvfit`: `5.213`
- tempered `qvfit`: `5.454`

So the broader validation kept the same ordering:

- hard-gated `qvfit` was best overall
- `vfit` stayed second
- tempered `qvfit` improved on raw `qvfit` but did not beat the gate
- raw `qvfit` remained the cautionary baseline

---

## Detailed Read

The gate behaved the way the mechanism predicted:

- on low-dispersion `attn_mass` supports, it stayed fully open and kept the
  real `qvfit` gains
- on high-dispersion `OMP` and `quotient_omit_omp` supports, it opened only
  about half the time and removed most of the large raw-`qvfit` losses
- on `repeat-prefill`, where some OMP-family supports were actually compatible,
  the gate still allowed enough `qvfit` use to improve over plain `vfit`

This is the key result:

- the gate is not just a defensive rollback to `vfit`
- it is a selective use policy that preserves real quotient-aware refit gains
  where the compatibility statistic says the solve is safe enough

The tempered solve is still useful as a mechanism check:

- it confirms that controlling row-scaling severity helps
- but the simple `gamma=0.5` power tempering did not outperform the hard gate
  on either slice

---

## Current Conclusion

The first genuine quotient-aware refit policy winner is now:

- hard-gated `qvfit` using `zhat_over_zref_cv`

That is important because it turns the quotient-refit story into a real
engineering policy:

- measure quotient row-scaling dispersion
- use `qvfit` only in the compatible regime

This is a materially stronger result than either:

- raw `qvfit` as a universal refit rule
- or a free blend-weight interpolation story

---

## Next Step

The next refit-side step should be:

1. treat hard-gated `qvfit` as the baseline policy to beat
2. test one sharper controlled variant than simple power tempering
   - clipped row scaling
   - quotient-ratio clipping
   - quotient-metric preconditioning
3. compare against:
   - `vfit`
   - raw `qvfit`
   - hard-gated `qvfit`

So the refit-side program has now progressed from:

- quotient mechanism

to:

- quotient compatibility statistic

to:

- quotient-aware control law
