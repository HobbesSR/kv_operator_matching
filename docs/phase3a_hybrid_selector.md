# Phase 3A: First Hybrid Support Selector

This note freezes the first Phase 3A result: a single continuous selector over
original-token candidates beats both anchor families on the tested evidence
surfaces.

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

## Conclusion

Phase 3A has already cleared more than the “bridge” bar.

The baseline-vs-repairability tradeoff from Phase 2 is not only descriptive; it
is a useful optimization principle. The first continuous hybrid selector is
already a real method candidate, not just a unification device.
