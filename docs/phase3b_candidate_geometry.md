# Phase 3B Precursor: Candidate Geometry Is Real but Not Yet Enough

This note freezes the cheap precursor to centroid-compatible local merging.

The experiment kept support as original tokens only and asked:

> does a locality-constrained operator-role compatibility score identify better
> local pair candidates than plain adjacency?

The score used:

- query-conditioned role similarity from the weighted mass-frame columns
- minus a small distance penalty inside a fixed local window

No merge construction was introduced. Candidate geometry was isolated from
representative fitting.

## Precommit

- [phase3b_candidate_geometry.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_candidate_geometry.md)

## Main Artifact

- [phase3b_candidate_geometry_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_candidate_geometry_q256_t32.json)

## Broad Result

The primary precommit failed:

- the top compatibility-scored neighbor did **not** beat the best adjacent
  neighbor on most anchors

Broad all-layer fractions:

- `online`: `top_compat_beats_adjacent_frac = 0.192`
- `teacher-forced`: `0.202`
- `repeat-prefill`: `0.192`

So the current compatibility score does not yet justify moving directly to a
centroid-compatible merge-construction tranche.

## Why The Result Is Still Useful

The diagnostic did reveal a real non-adjacent geometry signal:

- top compatibility neighbors beat a random local neighbor very often:
  - `online`: `0.848`
  - `teacher-forced`: `0.838`
  - `repeat-prefill`: `0.852`
- top compatibility neighbors were frequently non-adjacent:
  - `online`: `0.416`
  - `teacher-forced`: `0.444`
  - `repeat-prefill`: `0.441`
- compatibility score and local pair-fit loss were meaningfully aligned:
  - mean correlation about `-0.42` to `-0.43`

So the signal is not noise. It is just not strong enough, in its current form,
to displace adjacency as the top local partner on most anchors.

## Candidate Pool Read

The most practically important positive result is that the local candidate pool
is materially richer than the over-pruned fitted-pair tranche:

- mean eligible fraction versus adjacency:
  - `online`: `0.068`
  - `teacher-forced`: `0.073`
  - `repeat-prefill`: `0.073`

That is an order of magnitude larger than the fitted-pair tranche’s effective
merge usage (`~0.001-0.002`) and larger than its eligible-pair fraction
(`~0.003-0.005`).

So the candidate-geometry idea is real in one sense:

- it does expose a substantially larger non-adjacent local pool

But it is not yet real enough in another:

- it does not beat adjacency broadly as a top partner rule

## Conclusion

This precursor narrows the next move:

> local operator-role geometry is a real signal and materially richer than
> strict adjacency, but the current score is not strong enough to promote
> centroid-compatible local merging as-is.

So the next step should not be a full merge-construction tranche yet. It should
be one more cheap geometry iteration:

- improve the compatibility score itself
- then rerun the same diagnostic before building new constructed atoms
