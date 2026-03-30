# Phase 3B Precommit: Adjacent Pair-Merge Candidates

- `Hypothesis`: Adding a small source-grounded constructed-atom pool
  (original tokens plus adjacent pair merges, selected conflict-aware by the
  fixed `ΔB + ΔQ_coh` core) will improve `hybrid_pairmerge+vfit` over the
  original-token `hybrid+vfit` on richer decode-like surfaces
  (`teacher-forced`, `repeat-prefill`) while staying competitive online.
- `Surface`:
  broad surface = prompts `near_capacity_dispatch.json`,
  `near_capacity_dispatch_safe.json`, `near_capacity_network_dispatch_safe.json`,
  `relational_binding_probe.json`; layers `4 12 20 28`; budgets `0.25 0.5`;
  regimes `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
  dense follow-up = same prompts/regimes, layers `4 20`, budgets `0.25 0.5`,
  query-bank density `max_queries=512`, `max_new_tokens=64`.
- `Primary metric`: mean absolute holdout `L_true` for `+vfit` methods,
  with direct comparison `hybrid_pairmerge+vfit` versus `hybrid+vfit`.
- `Mechanism expectation`: if the constructed substrate helps, it should do so
  via a nontrivial merged-atom usage rate and strongest gains on richer
  surfaces rather than sparse online only.
- `Promotion rule`: promote the pair-merge substrate only if it beats the
  original-token hybrid on the broad surface in at least the richer regimes and
  does not collapse on the dense follow-up. A slice-only win is follow-up
  interest, not promotion.
- `Kill rule`: if pair-merge is worse than the original-token hybrid on the
  broad surface in all regimes, or only shows a narrow slice win while losing
  broadly, treat it as a failed first Phase 3B substrate.
- `Artifact path`: `results/scratch/phase3b_pairmerge_*.json`
- `Pivot condition / follow-up trigger`: if the main hypothesis fails but the
  broad surface shows systematic merged-atom use without quality gain, treat
  that as evidence the merge construction itself is the bottleneck and precommit
  a different construction family rather than relabeling this tranche a success.
