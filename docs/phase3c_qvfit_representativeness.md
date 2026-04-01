# Phase 3C: Representativeness-Controlled QVFit

The hard-gated `qvfit` tranche established a real control law:

- use the quotient row metric only when quotient row-scaling dispersion is tame

That worked, but it is still a binary policy.

This tranche tests a smoother version of the same principle:

- use as much quotient geometry as possible
- while keeping the bank representation sufficiently broad

So the control is no longer "clip rows because they look large."
It is:

- preserve minimum bank representativeness under the induced row metric

---

## Principle

Write the fixed-support refit family as

$$
\min_C \|W(XC - O)\|_F^2
$$

where:

- `W = I` gives neutral-metric `vfit`
- `W = D` gives quotient-metric `qvfit`

The question is how to choose an admissible `W` between those extremes.

This tranche tests two principled controls.

### Effective-Sample-Size Control

Normalize the induced row energies into a probability distribution `p`.
Choose the strongest quotient row metric whose effective sample size stays above
a fixed fraction of the neutral-metric baseline.

In this tranche:

- `N_eff(W) / N_eff(I) >= 0.5`

### Divergence-To-Neutral Control

Treat neutral `vfit` as the baseline bank geometry and quotient `qvfit` as a
distorted one.
Choose the strongest quotient row metric whose normalized row distribution stays
close enough to the neutral metric.

In this tranche:

- `KL(p_W || p_neutral) <= 0.25`

So these are not arbitrary blends. They are admissibility rules on the
row-metric family.

---

## Experiment

Entry point:

- [phase3c_qvfit_policy.py](/home/csmith/projects/kv_operator_matching/experiments/qwen2_online_nz_match/phase3c_qvfit_policy.py)

Implemented controls:

- hard-gated `qvfit`
- power-tempered `qvfit`
- ESS-controlled `qvfit`
- divergence-bounded `qvfit`

Supports were held fixed. The tranche only changes the refit policy.

---

## First Slice

Artifact:

- [phase3c_qvfit_policy_representative_first_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_policy_representative_first_slice.json)

Overall mean holdout `L_true` by refit policy family:

- `vfit`: `5.398`
- raw `qvfit`: `5.866`
- hard-gated `qvfit`: `5.329`
- divergence-bounded `qvfit`: `5.367`
- ESS-controlled `qvfit`: `5.385`
- tempered `qvfit`: `5.617`

So on the first slice:

- both representativeness-controlled variants beat plain `vfit`
- both clearly improved on raw `qvfit`
- but neither beat the hard gate overall

The interesting nuance is support-family sensitivity:

- `attn_mass`: full `qvfit` remained best
- `OMP` and `quotient_omit_omp`: hard gate still dominated
- `hybrid`: the smoother ESS / divergence controls beat both plain `vfit` and
  the hard gate in `online`

So the smoother controls are not useless middles. They are real policies with a
different tradeoff surface.

---

## Broader Validation

Artifact:

- [phase3c_qvfit_policy_representative_stability_slice.json](/home/csmith/projects/kv_operator_matching/results/scratch/phase3c_qvfit_policy_representative_stability_slice.json)

Overall mean holdout `L_true` on the broader slice:

- `vfit`: `5.275`
- raw `qvfit`: `5.698`
- hard-gated `qvfit`: `5.213`
- divergence-bounded `qvfit`: `5.239`
- ESS-controlled `qvfit`: `5.256`
- tempered `qvfit`: `5.454`

The ordering stayed stable:

- hard gate remained best overall
- both representativeness-controlled variants remained better than plain `vfit`
- both remained much better than raw `qvfit`
- divergence-bounded control was slightly better than ESS control on aggregate

Regime read:

- `online`: hard gate remained best overall
- `repeat-prefill`: ESS and divergence controls slightly beat the gate overall
- `teacher-forced-suffix`: hard gate remained best

Support-family read on the broader slice:

- `attn_mass`
  - full `qvfit` still wins
  - any control weakens the gain somewhat
- `OMP`
  - hard gate still best
  - smoother controls help relative to raw `qvfit` but do not beat the gate
- `quotient_omit_omp`
  - same pattern as `OMP`
- `hybrid`
  - divergence-bounded and ESS-controlled refits beat both plain `vfit` and
    the hard gate overall

That last point is the main new result.

---

## Interpretation

The hard gate is still the best general refit policy.

But the representativeness controls add a sharper structural lesson:

- binary admissibility is not the whole story
- some support families, especially `hybrid`, benefit from a controlled
  quotient metric even when the full quotient metric is too aggressive

So the new picture is:

- low-dispersion supports (`attn_mass`) want the full quotient metric
- high-dispersion supports (`OMP`, `quotient_omit_omp`) still want hard gating
- intermediate supports (`hybrid`) can benefit from a smoother
  representativeness-controlled quotient metric

That means the refit bottleneck is now better specified than "gate or not":

- different support families want different admissible strengths of quotient
  geometry

---

## Current Status

The refit-side ranking is now:

1. hard-gated `qvfit` as the best overall policy
2. divergence-bounded / ESS-controlled `qvfit` as real second-generation
   policies that recover some support families the gate leaves on the table
3. tempered `qvfit` as a weaker generic control
4. raw `qvfit` as the cautionary baseline

So the next step is no longer "invent another blend."
It is:

- learn or derive a policy that selects among
  - full quotient metric
  - hard gate
  - representativeness-controlled quotient metric

based on support-family compatibility diagnostics.
