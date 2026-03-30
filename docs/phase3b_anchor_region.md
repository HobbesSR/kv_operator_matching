# Phase 3B: Anchor-Conditioned Local Construction Failed Broadly

This note freezes the next Phase 3B substrate shift after the pairwise local
branches were demoted.

The construction kept the live Phase 3A selector fixed at `ΔB + ΔQ_coh` and
changed the substrate only:

- start from the selected hybrid anchors
- assign nearby tokens to anchors inside a fixed local window
- replace each selected anchor with one conservative regional representative

So this tested anchor-conditioned local construction rather than pairwise
partner search.

## Precommit

- [phase3b_anchor_region.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_anchor_region.md)

## Main Artifact

- [phase3b_anchor_region_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_anchor_region_q256_t32.json)

## Broad Result

This tranche failed the precommit. Against the original-token
`hybrid+vfit` baseline, the anchor-conditioned construction was worse on the
broad surface in every regime:

- `online`: `5.19` vs `4.04` (`+1.15`)
- `teacher-forced`: `5.07` vs `4.41` (`+0.67`)
- `repeat-prefill`: `9.54` vs `9.44` (`+0.10`)

It did not clear the promotion bar, so there is no dense follow-up.

## Why The Failure Is Still Useful

This was not a no-op or a revert to the original-token support:

- mean changed-atom fraction was high:
  - `online`: `0.618`
  - `teacher-forced`: `0.650`
  - `repeat-prefill`: `0.644`
- mean assigned-token fraction was very high:
  - `online`: `0.959`
  - `teacher-forced`: `0.959`
  - `repeat-prefill`: `0.970`
- mean region size was about `2.85-2.89`

So the construction was heavily used. The failure is in the local regional
construction family itself, not in the selector refusing to leave raw tokens.

## Mechanism Read

There is one limited positive signal:

- before `vfit`, the regional construction helped under `repeat-prefill`
  (`11.29` vs `12.21`)

But that did not survive the full method path:

- after `vfit`, even `repeat-prefill` became slightly worse (`9.54` vs `9.44`)

So this family can modestly improve baseline support in a supervision-rich
control regime, but it does not preserve the repairability that made the
original-token hybrid support strong.

## Conclusion

This tranche prunes another tempting story:

> building one conservative regional representative around each good anchor is
> not enough to beat the original-token hybrid support.

Together with the pairwise failures, that means simple local constructed-atom
families now look weak enough that the next Phase 3B move should be a larger
framing shift, not another small local construction variant.
