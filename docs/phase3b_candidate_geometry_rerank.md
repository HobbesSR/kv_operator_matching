# Phase 3B Precursor Follow-Up: Locality-Regularized Rerank Still Fails

This note freezes the next cheap geometry iteration after
[phase3b_candidate_geometry.md](/home/csmith/projects/kv_operator_matching/docs/phase3b_candidate_geometry.md).

The question was narrower than another merge tranche:

> if adjacency is a very strong local prior, can a more locality-regularized
> compatibility score beat the current score on the top-neighbor metric without
> collapsing back to adjacency-only behavior?

The support remained original tokens only. No merge construction or
representative fitting was introduced.

## Precommit

- [phase3b_candidate_geometry_rerank.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_candidate_geometry_rerank.md)

## Main Artifact

- [phase3b_candidate_geometry_rerank_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_candidate_geometry_rerank_q256_t32.json)

## Score Comparison

The rerank compared two scores on the same local candidate pool:

- `current`: the existing role-similarity score with a mild distance penalty
- `locality_regularized`: a stronger locality-regularized score that leaves
  adjacency unpenalized and penalizes farther hops more sharply

The primary metric was compared *within this same run*. That matters because
the rerank script recomputed the adjacent-loss reference internally, so the
absolute `beats_adjacent` level should be compared between the two scores here,
not numerically against the earlier artifact.

## Broad Result

The precommit failed clearly. The locality-regularized rerank was worse than
the current score on the top-neighbor metric in every regime:

- `online`:
  - `current`: `top_score_beats_adjacent_frac = 0.153`
  - `locality_regularized`: `0.113`
- `teacher-forced`:
  - `current`: `0.163`
  - `locality_regularized`: `0.121`
- `repeat-prefill`:
  - `current`: `0.152`
  - `locality_regularized`: `0.115`

So the additional locality prior moved the chosen partner closer, but it did
not improve top-neighbor quality relative to adjacency.

## Mechanism Read

The rerank did what it was supposed to do geometrically:

- it chose closer neighbors
  - `online`: mean top-partner distance `2.27 -> 1.72`
  - `teacher-forced`: `2.39 -> 1.81`
  - `repeat-prefill`: `2.29 -> 1.78`
- it reduced non-adjacent selections
  - `online`: `0.416 -> 0.293`
  - `teacher-forced`: `0.444 -> 0.317`
  - `repeat-prefill`: `0.441 -> 0.323`

But it did **not** improve the actual partner quality:

- the “beats random local neighbor” property stayed strong but mostly flat
- the eligible pool shrank materially
  - `online`: `0.068 -> 0.037`
  - `teacher-forced`: `0.073 -> 0.041`
  - `repeat-prefill`: `0.073 -> 0.043`

So this was not a better compatibility signal. It was mostly a retreat toward
adjacency with less candidate richness.

## Conclusion

This follow-up closes the cheap score-tweak branch:

> stronger locality regularization does not rescue the local compatibility
> score as a top partner rule.

The broader candidate-geometry idea remains real, but the current local-pair
framing still does not beat adjacency where it matters most.

That means the next step should not be another small scoring tweak. It should
reconsider the local-pair framing itself before another merge-construction
tranche.
