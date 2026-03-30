# Phase 3B Follow-up Precommit: Mid-Layer Rich-Regime Slice

- `Hypothesis`: The adjacent pair-merge substrate is only viable in richer
  regimes and mid layers; on a denser `teacher-forced` / `repeat-prefill`
  slice around layers `12` and `20`, `hybrid_pairmerge+vfit` will improve over
  the original-token `hybrid+vfit`.
- `Surface`: prompts `near_capacity_dispatch.json`,
  `near_capacity_dispatch_safe.json`, `near_capacity_network_dispatch_safe.json`,
  `relational_binding_probe.json`; layers `12 20`; budgets `0.25 0.5`;
  regimes `teacher-forced repeat-prefill`; query-bank density
  `max_queries=512`, `max_new_tokens=64`.
- `Primary metric`: mean absolute holdout `L_true` for
  `hybrid_pairmerge+vfit` versus `hybrid+vfit`.
- `Mechanism expectation`: if the slice lead is real, merged-atom usage should
  stay nontrivial and the advantage should concentrate in the richer regimes
  rather than flipping back to the original-token hybrid at higher density.
- `Promotion rule`: this can at most promote a slice-level follow-up claim, not
  the broad Phase 3B substrate. It only counts as a real slice lead if both
  richer regimes improve on average.
- `Kill rule`: if the denser slice does not improve in both richer regimes,
  treat adjacent pair-merge as an unhelpful first construction family and move
  on to a different construction rule.
- `Artifact path`: `results/scratch/phase3b_pairmerge_q512_t64_l1220_tf_rp.json`
