# Phase 3B Precursor: Centroid-Conditioned Regional Assignment Failed

This note freezes the centroid/operator-role assignment precursor that followed
the failed locality-first Phase 3B constructions.

The comparison held two things fixed:

- the live Phase 3A hybrid anchors
- the same conservative regional representative construction

and changed only the assignment rule:

- `anchor_region_local`: assign nearby tokens to anchors using the existing
  local role score
- `anchor_region_centroid`: assign nearby tokens to anchors using a
  centroid-conditioned operator-role fingerprint

So this was a clean test of grouping principle rather than a new construction
family.

## Precommit

- [phase3b_centroid_region_precursor.md](/home/csmith/projects/kv_operator_matching/docs/precommits/phase3b_centroid_region_precursor.md)

## Main Artifact

- [phase3b_centroid_region_precursor_q256_t32.json](/home/csmith/projects/kv_operator_matching/results/checkpoints/phase3/phase3b_centroid_region_precursor_q256_t32.json)

## Broad Result

The precommit failed. Centroid-conditioned assignment was worse than
locality-only assignment after `vfit` in all three regimes:

- `online`: `4.49` vs `3.88` (`+0.61`)
- `teacher-forced`: `4.88` vs `4.44` (`+0.44`)
- `repeat-prefill`: `9.82` vs `9.34` (`+0.48`)

It improved only `8/64` online cells, `15/64` teacher-forced cells, and
`15/64` repeat-prefill cells.

So simple centroid-conditioned regional assignment does **not** beat the
locality-first regional assignment as a basis for value repair.

## Mechanism Read

The interesting part is that the mechanism story is mixed rather than empty.

Centroid assignment did improve internal assignment coherence:

- mean assignment-similarity delta:
  - `online`: `+0.205`
  - `teacher-forced`: `+0.211`
  - `repeat-prefill`: `+0.199`

It also slightly reduced post-`vfit` low-singular update share:

- mean low-sv delta-share delta:
  - `online`: `-0.015`
  - `teacher-forced`: `-0.013`
  - `repeat-prefill`: `-0.022`

But it hurt the construction geometry more strongly:

- mean design stable-rank delta:
  - `online`: `-0.231`
  - `teacher-forced`: `-0.452`
  - `repeat-prefill`: `-1.300`

So the centroid-conditioned groups are more internally similar, but they are
still a worse repair substrate overall.

## Conclusion

This is a useful negative result:

> better operator-role coherence inside local regions is not enough, by itself,
> to preserve the repairability advantages of the original-token hybrid
> support.

That means the next Phase 3B move should not be a straightforward promotion of
centroid-conditioned regional construction. The missing ingredient is still not
just “better grouping by role similarity” in this simple anchor-region form.
