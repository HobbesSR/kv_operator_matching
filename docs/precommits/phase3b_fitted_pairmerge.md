# Phase 3B Precommit: Compatibility-Filtered Fitted Pair Representatives

- `Hypothesis`: Replacing arithmetic adjacent-pair means with conservative,
  compatibility-filtered, locally fitted pair representatives will improve over
  mean-merges broadly and may beat the original-token `hybrid+vfit` in richer
  regimes without collapsing online.
- `Surface`:
  broad surface = prompts `near_capacity_dispatch.json`,
  `near_capacity_dispatch_safe.json`, `near_capacity_network_dispatch_safe.json`,
  `relational_binding_probe.json`; layers `4 12 20 28`; budgets `0.25 0.5`;
  regimes `online teacher-forced repeat-prefill`; query-bank density
  `max_queries=256`, `max_new_tokens=32`.
  dense follow-up = same prompts/regimes, layers `12 20`, budgets `0.25 0.5`,
  query-bank density `max_queries=512`, `max_new_tokens=64`.
- `Primary metric`: mean absolute holdout `L_true` for
  `hybrid_pairmerge_fitted+vfit` versus both `hybrid_pairmerge_mean+vfit` and
  `hybrid+vfit`.
- `Mechanism expectation`: fitted-pair candidates should beat mean-merges on
  baseline `L_true`, maintain nontrivial merged-atom selection, and expose an
  inspectable eligible-pair fraction so failures can be attributed to filter
  admission versus representative quality.
- `Promotion rule`: promote fitted pair representatives only if they beat
  mean-merges broadly and at least beat the original-token hybrid on the broad
  surface in richer regimes or survive the dense follow-up as a clear slice
  lead.
- `Kill rule`: if fitted pair representatives fail to beat mean-merges broadly
  or still lose cleanly to the original-token hybrid on both the broad surface
  and dense follow-up, demote simple pairwise merge as the main Phase 3B path.
- `Artifact path`: `results/scratch/phase3b_fitted_pairmerge_*.json`
