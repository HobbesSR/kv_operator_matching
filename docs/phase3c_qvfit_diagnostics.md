# Phase 3C: QVFit Compatibility Diagnostics

The fixed-support refit tranche established a real but conditional result:

- `qvfit` helps some supports
- `qvfit` hurts others

So the next question is not whether quotient-aware refit matters at all. It
does. The question is what predicts whether it helps or destabilizes.

This diagnostic tranche inspects the geometry of the same fixed supports under
two design views:

- ordinary `vfit` design: normalized compact attention weights
- quotient-aware `qvfit` design: unnormalized compact numerator weights

The goal is to see whether `qvfit` failure is mostly about:

- support family label
- ordinary design conditioning
- or the extra row-scaling introduced by the quotient-weighted solve

---

## Diagnostic Idea

For fixed support and fixed `beta`, ordinary `vfit` uses the design:

$$
\alpha(q) \sqrt{w_q}
$$

while `qvfit` uses:

$$
\hat Z(q)\alpha(q) \sqrt{w_q}.
$$

So `qvfit` does not just change the target. It also rescales rows by the
compact mass `\hat Z(q)`.

This immediately suggests compatibility diagnostics:

- condition number of the ordinary design
- condition number of the quotient-weighted design
- concentration of quotient-weighted row energy
- variation in `\hat Z(q) / Z(q)`
- baseline quotient-cancellation statistics
- beta spread on the fixed support

---

## Experiment

Entry point:

- [phase3c_qvfit_diagnostics.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_qvfit_diagnostics.py)

Artifact:

- [phase3c_qvfit_diagnostics_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_diagnostics_first_slice.json)

Command:

```bash
python experiments/qwen2_online_nz_match/phase3c_qvfit_diagnostics.py \
  --layers 4 20 \
  --budgets 0.25 0.5 \
  --collection-modes online teacher-forced-suffix repeat-prefill \
  --prompt-files near_capacity_dispatch_safe.json relational_binding_probe.json \
  --max-queries 256 \
  --max-new-tokens 32 \
  --save-json results/scratch/phase3c_qvfit_diagnostics_first_slice.json
```

---

## First Result

The strongest finding is that `qvfit` success tracks the quotient-weighted row
scaling much more than the ordinary normalized-design geometry.

Global correlations with `qvfit_minus_vfit_l_true`:

- `alpha_design_condition_number`: `-0.02`
- `q_design_condition_number`: `+0.40`
- `q_row_energy_top5_share`: `+0.59`
- `q_row_energy_cv`: `+0.53`
- `zhat_over_zref_cv`: `+0.85`
- `base_train_qr_cancellation_gain`: `+0.40`
- `base_train_qr_worst_ratio`: `+0.58`
- `beta_cv`: `+0.65`

The cleanest signal is `zhat_over_zref_cv`.
When the compact-to-reference mass ratio varies strongly across the bank,
`qvfit` is much more likely to be worse than ordinary `vfit`.

Method-level read:

- `attn_mass`
  - low `zhat_over_zref_cv`
  - low quotient-row concentration
  - `qvfit` helps in all three regimes
- `OMP` and `quotient_omit_omp`
  - much higher `zhat_over_zref_cv`
  - much worse quotient-design conditioning
  - `qvfit` usually hurts, especially in `online` and
    `teacher-forced-suffix`
- `hybrid`
  - intermediate but still often unstable under `qvfit`
  - especially where quotient-row concentration is elevated

This is important because it points to a concrete mechanism:

- `qvfit` is not mainly failing because the ordinary support design is bad
- it is failing because the quotient-weighted solve over-concentrates some bank
  rows and amplifies mass-ratio variation

---

## Interpretation

The first compatibility result is:

- ordinary `vfit` design conditioning by itself does not explain `qvfit`
  success or failure well
- quotient-specific row-scaling diagnostics do

So the next refit-side problem is not "find another support family at random."
It is:

- control quotient row-scaling concentration
- or use quotient-aware refit only when the bank/support pair passes a
  compatibility check

This also sharpens the Phase 3C steering sentence:

- the bottleneck is finding support and representation classes where
  quotient-aware fitting is stable and useful

because the current failure mode is now measurable rather than abstract.

---

## Next Experiment

The next principled follow-up is not a free blend weight. It is one of:

1. a gated `qvfit` rule:
   only use `qvfit` when `zhat_over_zref_cv` and quotient-row concentration are
   below a threshold

2. a coverage-clipped quotient refit:
   keep the quotient-aware objective, but cap or regularize the row scaling
   driven by `\hat Z(q)`

3. a support-side compatibility screen:
   use quotient-aware refit only with support families whose quotient-weighted
   design stays well behaved

That is a much tighter next step than another blind sweep over support labels.

One cheap offline check already supports that direction.
Using the diagnostic artifact itself, a simple gate that chooses `qvfit` only
when `zhat_over_zref_cv <= 0.25` would have improved the first-slice overall
mean holdout `L_true` from `5.398` (`always vfit`) to `5.329`, while avoiding
most of the large `OMP` / `quotient_omit_omp` failures.

So the next experiment is no longer speculative in the abstract.
There is already evidence that a structural compatibility gate could turn the
diagnostic result into a better refit policy.

---

## Broader Validation

Broader artifact:

- [phase3c_qvfit_diagnostics_stability_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_diagnostics_stability_slice.json)

Setup matched the broader shortlist-stability surface:

- prompts: `near_capacity_dispatch`, `near_capacity_dispatch_safe`,
  `near_capacity_network_dispatch_safe`, `relational_binding_probe`
- layers: `4`, `20`
- budgets: `0.25`, `0.5`
- regimes: `online`, `teacher-forced-suffix`, `repeat-prefill`

The broader slice preserved the main mechanism almost intact:

- `alpha_design_condition_number`: `-0.02`
- `q_design_condition_number`: `+0.44`
- `q_row_energy_top5_share`: `+0.53`
- `zhat_over_zref_cv`: `+0.81`

So `zhat_over_zref_cv` remained the dominant single predictor of
`qvfit - vfit`, even after broadening the prompt surface.

The cheap gate simulation also survived:

- `always vfit`: `5.275`
- `always qvfit`: `5.698`
- `qvfit` only when `zhat_over_zref_cv <= 0.25`: `5.213`

The exact best threshold is still provisional, but the mechanism and gate form
now look much more robust than a first-slice artifact.
