# Phase 3B: First Constructed-Support Tranche Failed

This note freezes the first Phase 3B substrate experiment: a conflict-aware
hybrid selector over a small source-grounded candidate pool consisting of:

- original-token singleton atoms
- adjacent pair merges `(i, i+1)` with simple mean keys and values

The selector core remained fixed at the stable Phase 3A objective:

`ΔB + ΔQ_coh`

The question was whether a modest constructed-atom pool could improve the
original-token hybrid selector before introducing tree structure or more
ambitious synthetic atoms.

## Precommits

- Broad tranche:
  [phase3b_pairmerge.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_pairmerge.md)
- Slice follow-up:
  [phase3b_pairmerge_slice_followup.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_pairmerge_slice_followup.md)

## Main Artifacts

- Broad surface:
  [phase3b_pairmerge_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_pairmerge_q256_t32.json)
- Dense slice follow-up:
  [phase3b_pairmerge_q512_t64_l1220_tf_rp.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_pairmerge_q512_t64_l1220_tf_rp.json)

## Broad Result

The broad precommit failed. Against the original-token hybrid baseline:

- `online`:
  - `hybrid+vfit`: `4.04`
  - `hybrid_pairmerge+vfit`: `6.56`
  - mean delta: `+2.52`
- `teacher-forced`:
  - `hybrid+vfit`: `4.41`
  - `hybrid_pairmerge+vfit`: `4.95`
  - mean delta: `+0.55`
- `repeat-prefill`:
  - `hybrid+vfit`: `9.22`
  - `hybrid_pairmerge+vfit`: `9.22`
  - mean delta: `-0.004`

So the first constructed substrate did **not** beat the original-token hybrid
on the broad surface. It was substantially worse online, worse teacher-forced,
and only tie-level on repeat-prefill.

## Mechanism Read

This was not a null experiment. The selector did use merged atoms:

- broad mean merged fraction:
  - `online`: `0.227`
  - `teacher-forced`: `0.214`
  - `repeat-prefill`: `0.237`

So the result is not “merges were never selected.” The constructed atom family
was used materially, but did not improve quality. That points at the merge
construction itself as the bottleneck, not mere selector conservatism.

## Slice Follow-up

The broad run showed a small mid-layer richer-regime lead, so a second
precommitted slice tested layers `12` and `20` under denser
`teacher-forced` / `repeat-prefill` evidence.

That also failed:

- `teacher-forced`:
  - `hybrid+vfit`: `3.55`
  - `hybrid_pairmerge+vfit`: `4.54`
  - mean delta: `+0.99`
- `repeat-prefill`:
  - `hybrid+vfit`: `8.48`
  - `hybrid_pairmerge+vfit`: `9.26`
  - mean delta: `+0.78`

The denser slice therefore killed the “mid-layer richer-regime rescue” idea.

## Conclusion

The first Phase 3B construction family should be treated as a negative result:

> Adjacent pair-merge candidates are a live substrate shift, not a no-op, but
> this simple source-grounded merge construction does not improve the stable
> Phase 3A selector and should not be promoted.

The next Phase 3B construction family should therefore be different in kind,
not a minor variation on adjacent pair means.
