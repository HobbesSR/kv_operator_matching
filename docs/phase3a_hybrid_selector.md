# Phase 3A: First Hybrid Support Selector

This note freezes the first Phase 3A result and the immediate forensic
correction. A single continuous selector over original-token candidates is a
real method candidate, but the mechanism story is narrower than the first
checkpoint suggested.

## Selector

The first selector implements the frozen roadmap target:

`J_add = ΔB + alpha(E) * ΔQ_coh - beta(E) * ΔQ_span`

with:

- `ΔB`: incremental baseline-fidelity gain on the stable mass frame
- `ΔQ_coh`: stable-rank-like novelty term
- `ΔQ_span`: increase in normalized temporal support span
- `alpha(E), beta(E)`: continuous weights from query-bank observables already
  available in the harness

Constraints respected in this first version:

- one selector, no hand-switched regimes
- original-token candidates only
- a single coherence term and a single span term

## Main Artifacts

- Broad comparison surface:
  [phase3a_hybrid_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_hybrid_q256_t32.json)
- Denser confirmation surface:
  [phase3a_hybrid_q512_t64_l420.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_hybrid_q512_t64_l420.json)
- Support-profile inspection:
  [phase3a_hybrid_support_profile_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_hybrid_support_profile_q256_t32.json)
- Broad ablations and geometry:
  [phase3a_ablation_forensics_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_ablation_forensics_q256_t32.json)
- Low-singular-risk replacement check:
  [phase3a_ablation_forensics_lowsv_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_ablation_forensics_lowsv_q256_t32.json)
- Reduced stress slice:
  [phase3a_hybrid_stress_q384_t48_l2028.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3a_hybrid_stress_q384_t48_l2028.json)

## Headline Result

On the broad `q256/t32` surface, `hybrid+vfit` beat both `recency+vfit` and
`omp+vfit` in mean absolute holdout `L_true`:

- `online`: `4.16` vs `4.98` vs `5.52`
- `teacher-forced`: `4.19` vs `6.74` vs `5.63`
- `repeat-prefill`: `9.34` vs `11.84` vs `9.99`

The hybrid baseline itself was also strongest:

- `online`: `hybrid 6.10`, `recency 7.74`, `omp 8.87`
- `teacher-forced`: `6.34`, `12.64`, `9.80`
- `repeat-prefill`: `11.95`, `17.00`, `12.92`

The denser `q512/t64` follow-up preserved the same ordering on the tested
layers (`4`, `20`):

- `online`: `hybrid+vfit 2.14`, `recency+vfit 2.65`, `omp+vfit 4.53`
- `teacher-forced`: `2.72`, `4.16`, `3.29`
- `repeat-prefill`: `7.70`, `9.38`, `8.97`

## Structural Read

The support-profile pass shows the selector is not a disguised regime switch.

Average support overlap:

- `online`: recency `0.442`, OMP `0.534`
- `teacher-forced`: recency `0.389`, OMP `0.530`
- `repeat-prefill`: recency `0.374`, OMP `0.560`

So the selector does move continuously with evidence state:

- thinner evidence shifts it somewhat toward recency
- richer evidence shifts it somewhat toward OMP

But the stronger finding is that it mostly occupies a distinct hybrid region
and still outperforms both anchors.

## Forensic Correction

The first checkpoint overclaimed the mechanism slightly. The follow-up
forensics say:

- the current result is not mainly a recency-style locality story
- the effective payload is `ΔB + ΔQ_coh`
- the current span penalty is not clearly helping
- the current evidence-dependent weighting is not clearly helping

The clearest ablation result is that `ΔB` alone is not enough. Full hybrid
beats `ΔB`-only by a large margin in absolute held-out `L_true`:

- `online`: `-2.81`
- `teacher-forced`: `-1.62`
- `repeat-prefill`: `-1.51`

But `ΔB + ΔQ_coh` is essentially tied with the full score on the broad
surface:

- `online`: full minus `ΔB+ΔQ_coh` = `+0.12`
- `teacher-forced`: `-0.22`
- `repeat-prefill`: `+0.14`

This is strong enough to promote `ΔB + ΔQ_coh` as the live Phase 3A selector
core.

The next direct replacement test also failed cleanly: replacing raw span with a
low-singular-direction risk penalty made the selector much worse than both the
core and the old full score.

Broad `q256/t32` mean holdout `L_true` for the low-singular-risk variant:

- `online`: `6.61`
- `teacher-forced`: `6.40`
- `repeat-prefill`: `12.06`

Against the `ΔB + ΔQ_coh` core, that was worse by:

- `online`: `+2.57`
- `teacher-forced`: `+1.99`
- `repeat-prefill`: `+2.77`

The geometry pass also sharpens the mechanism story. Hybrid does **not** keep
recency-like support geometry:

- it remains broad (`span_frac` about `0.99+`)
- it is only moderately adjacent (`0.40-0.45`)
- its stable rank is below recency and close to OMP
- its low-singular update share is also much closer to OMP than recency

So the current best explanation is:

- hybrid wins primarily by producing a much better baseline support than
  recency and OMP
- then repairing slightly better than OMP
- not by preserving recency-style local coherence

## Stress Read

On the reduced stress surface (`layers 20,28`, budgets `12.5/25/50%`, denser
bank, plus `network_cutover` where possible), hybrid still robustly beat
recency but no longer beat OMP uniformly:

- `online`: `hybrid+vfit 6.85`, `recency+vfit 9.97`, `omp+vfit 7.29`
- `teacher-forced`: `7.86`, `13.62`, `7.18`
- `repeat-prefill`: `18.11`, `22.10`, `16.45`

## Conclusion

Phase 3A still clears the “bridge” bar, but the defensible conclusion is now
sharper:

> A continuous support objective driven primarily by baseline-fidelity gain
> plus coherence novelty yields a hybrid support family that adapts with
> evidence state, robustly outperforms recency, and can outperform OMP on
> sparse online surfaces; in the current formulation, the live core is
> `ΔB + ΔQ_coh`, while the span penalty, evidence-dependent weighting, and
> direct low-singular-risk replacement do not yet clearly contribute.
