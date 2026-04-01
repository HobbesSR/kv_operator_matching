# Phase 3C: Diagnostic-Conditioned QFit Controller

The previous refit tranches established three things:

- hard-gated `qvfit` is the best universal default
- smoother admissibility controls are real
- different support families prefer different admissible strengths of quotient
  geometry

That suggested a cleaner endpoint than maintaining a menu of fixed policies:

- one measured `qfit` controller that chooses the row metric from diagnostics

This tranche implements exactly that.

---

## Controller

The controller keeps the same fixed-support weighted least-squares family:

$$
\min_C \|W(XC - O)\|_F^2
$$

but chooses the row metric from three diagnostic branches.

Policy:

1. **Full quotient metric** if the full quotient geometry is already close
   enough to the neutral bank metric.
2. **Neutral fallback** if quotient row-scaling dispersion is clearly too high.
3. **KL-controlled middle metric** otherwise.

In the current tranche, the controller uses:

- full quotient if `q_weight_kl_to_neutral <= 0.9`
- neutral fallback if `zhat_over_zref_cv >= 0.5`
- otherwise use the strongest quotient metric with
  `KL(p_W || p_neutral) <= 0.25`

So this is not a family lookup table.
It is a measured law over the same `qfit` metric family.

---

## Experiment

Entry point:

- [phase3c_qvfit_policy.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_qvfit_policy.py)

The new method is:

- `qfit_diag`

Compared against:

- `vfit`
- hard-gated `qvfit`
- pure KL-controlled `qvfit`

---

## First Slice

Artifact:

- [phase3c_qfit_diag_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qfit_diag_first_slice.json)

Overall mean holdout `L_true`:

- `vfit`: `5.3980`
- hard-gated `qvfit`: `5.3294`
- KL-controlled `qvfit`: `5.3665`
- `qfit_diag`: `5.3189`

So on the first slice:

- `qfit_diag` beat the hard gate
- it also beat the pure KL-controlled policy

Family read:

- `attn_mass`
  - hard gate stayed best because it already collapses to full quotient
- `hybrid`
  - `qfit_diag` beat both the hard gate and pure KL control
- `OMP`
  - `qfit_diag` was essentially tied with the hard gate
- `quotient_omit_omp`
  - `qfit_diag` slightly beat the hard gate

This is the first result where the measured controller itself wins, not just
one of its component policy families.

---

## Broader Validation

Artifact:

- [phase3c_qfit_diag_stability_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qfit_diag_stability_slice.json)

Overall mean holdout `L_true`:

- `vfit`: `5.2748`
- hard-gated `qvfit`: `5.2125`
- KL-controlled `qvfit`: `5.2387`
- `qfit_diag`: `5.1943`

So the broader slice preserved the win.

By regime:

- `online`
  - `qfit_diag`: `3.2728`
  - hard gate: `3.2867`
- `repeat-prefill`
  - `qfit_diag`: `9.1639`
  - hard gate: `9.1913`
- `teacher-forced-suffix`
  - `qfit_diag`: `3.1462`
  - hard gate: `3.1595`

The controller is not making one dramatic regime trade.
It is producing small, consistent improvements across all three regimes.

Branch rates on the broader slice:

- `full_quotient`: `33.9%`
- `middle_kl`: `46.9%`
- `neutral_fallback`: `19.3%`

So the controller is genuinely using all three branches on the wider surface.

---

## Family Read

On the broader slice:

- `attn_mass`
  - hard gate still wins because it already stays close to full quotient
- `OMP`
  - `qfit_diag` slightly beats the hard gate
- `hybrid`
  - `qfit_diag` beats both the hard gate and pure KL control
- `quotient_omit_omp`
  - `qfit_diag` beats the hard gate

That means the controller is not winning by discovering one new support-family
truth.
It is winning by making fewer avoidable mistakes across the mixed surface.

---

## Interpretation

This is the cleanest refit-policy result so far.

The program has now progressed from:

- quotient mechanism
- quotient compatibility statistic
- hard-gated `qvfit`

to:

- a measured `qfit` controller that beats the hard gate on both the first slice
  and the broader validation surface

The win is modest, but it is structurally important:

- it validates the idea that the metric should be derived from measured bank
  state rather than selected from a fixed method menu
- it turns the refit side into a single adaptive family with `vfit` and
  `qvfit` as limiting cases

---

## Current Status

The best current refit policy is now:

- `qfit_diag`

with hard-gated `qvfit` demoted from "best policy" to "best simple universal
baseline."

The next question is no longer whether state-dependent metric control works.
It does.

The next question is whether the controller can be made more principled and
less threshold-like without losing the current gain, for example by:

- replacing the discrete branch rule with a continuous metric law
- deriving the admissibility thresholds from the diagnostics rather than
  setting them by hand
- or folding shortlist-side diagnostics into the same controller
