# Phase 3B: Fitted Pair Representatives Over-Pruned

This note freezes the second Phase 3B tranche: compatibility-filtered,
locally fitted adjacent-pair representatives.

The question was narrower than the first pair-mean failure:

> Can a conservative local representative construction rescue pairwise merging
> without destroying the repairability advantages of the original-token hybrid?

## Precommit

- Broad tranche:
  [phase3b_fitted_pairmerge.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_fitted_pairmerge.md)

## Main Artifact

- Broad surface:
  [phase3b_fitted_pairmerge_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_fitted_pairmerge_q256_t32.json)

## Broad Result

The fitted-pair tranche did **not** beat the original-token hybrid broadly.

Against `hybrid+vfit`:

- `online`:
  - `hybrid+vfit`: `4.04`
  - `hybrid_pairmerge_fitted+vfit`: `4.05`
  - mean delta: `+0.017`
- `teacher-forced`:
  - `hybrid+vfit`: `4.41`
  - `hybrid_pairmerge_fitted+vfit`: `4.36`
  - mean delta: `-0.045`
- `repeat-prefill`:
  - `hybrid+vfit`: `9.40`
  - `hybrid_pairmerge_fitted+vfit`: `9.46`
  - mean delta: `+0.058`

So this construction was basically tied with the original-token hybrid, not a
real substrate win.

## Why This Is Still Useful

The tranche **did** beat the failed mean-merge family:

- versus `hybrid_pairmerge_mean+vfit`, mean delta was:
  - `online`: `-2.51`
  - `teacher-forced`: `-0.59`
  - `repeat-prefill`: `-0.20`

But the mechanism read shows why: the compatibility filter admitted almost no
pairs, so the fitted construction mostly collapsed back to the original-token
selector.

Broad averages:

- eligible pair fraction:
  - `online`: `0.0035`
  - `teacher-forced`: `0.0033`
  - `repeat-prefill`: `0.0053`
- selected merged fraction:
  - `online`: `0.0013`
  - `teacher-forced`: `0.0010`
  - `repeat-prefill`: `0.0022`

That is the main result. This tranche did not show that fitted pair
representatives are good. It showed that this conservative compatibility filter
over-pruned the candidate pool so hard that the method mostly reverted to the
original-token hybrid.

## Conclusion

The fitted-pair tranche should be treated as another negative but informative
result:

> Conservative local fitting can rescue the mean-merge failure numerically, but
> with this compatibility filter it does so mainly by almost never merging.

So the next Phase 3B move should not be “more of the same with tiny threshold
tweaks.” It should change one of the two ingredients explicitly:

- a less over-pruned compatibility rule
- or a different construction family altogether

Given the cost of this tranche and the near-no-op merged usage, there was no
reason to run the dense follow-up.
